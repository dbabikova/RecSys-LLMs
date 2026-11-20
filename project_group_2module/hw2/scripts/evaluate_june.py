"""Evaluate frozen models on the exact shared June cohort, without retraining."""
import csv
import io
import json
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from input_contract import canonical,digest
from ranker_metrics import evaluate
from train_lambdamart import save_csv

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'data/hw2/june_evaluation'
    out.mkdir(exist_ok=True)
    frozen=json.loads((ROOT/'hw2/MODEL_FREEZE.json').read_text())
    model_path=ROOT/frozen['primary_model_private_path']
    encoder_path=ROOT/frozen['encoder_private_path']
    assert digest(model_path.read_bytes())==frozen['primary_model_sha256']
    assert digest(encoder_path.read_bytes())==frozen['encoder_sha256']
    payload_path=ROOT/'data/hw2/common_inputs/payloads.jsonl'
    manifest=json.loads((ROOT/'hw2/INPUTS.json').read_text())
    payloads=[json.loads(s) for s in payload_path.read_text().splitlines()]
    mappings=[json.loads(s) for s in (ROOT/'data/hw2/common_inputs/private_mapping.jsonl').read_text().splitlines()]
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/team5_alina_t4_protocol_check.zip') as z:
        protocol=json.loads(z.read('protocol.json'))
        shared=json.loads(z.read('metrics.json'))['methods']
    assert [r['user_id'] for r in mappings]==protocol['users']
    assert [p['user_alias'] for p in payloads]==[r['user_alias'] for r in mappings]
    truth=defaultdict(set)
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/FAR-Trans.zip') as z:
        raw=z.read('FAR-Trans/transactions.csv')
    sources=json.loads((ROOT/'hw2/SOURCES.json').read_text())
    assert digest(raw)==sources['source_csv_sha256']['transactions.csv']
    eligible=set(protocol['eligible_assets']); users=set(protocol['users'])
    for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
        if (r['customerID'] in users and int(r['transactionID'])>=0 and r['transactionType']=='Buy'
            and protocol['cutoff']<=r['timestamp'][:10]<protocol['end_exclusive'] and r['ISIN'] in eligible):
            truth[r['customerID']].add(r['ISIN'])
    groups=[]; poolmaps=[]
    for p,m in zip(payloads,mappings):
        aliases=[c['asset_alias'] for c in p['candidates']]
        ids=[m['asset_alias_to_isin'][a] for a in aliases]
        t=truth[m['user_id']]
        groups.append({'group_key':p['user_alias'],'labels':[int(i in t) for i in ids],
                       'full_truth_count':len(t),'candidate_aliases':aliases})
        poolmaps.append({'group_key':p['user_alias'],'pool_isins':ids})
    encoder=joblib.load(encoder_path)
    start=time.perf_counter()
    x=encoder.transform(payloads)
    model=lgb.Booster(model_file=str(model_path))
    scores=model.predict(x,num_threads=4)
    elapsed=time.perf_counter()-start
    np.save(out/'lambdamart_scores.npy',scores,allow_pickle=False)
    metric,rows,preds=evaluate(scores,groups,poolmaps)
    save_csv(out/'lambdamart_per_user.csv',rows)
    save_csv(out/'lambdamart_top10.csv',preds)
    # Independently recompute each value directly from saved ISIN rankings/full truth.
    for i,(p,m,row) in enumerate(zip(payloads,mappings,rows)):
        ranking=[r['ISIN'] for r in preds[i*10:(i+1)*10]]
        t=truth[m['user_id']]
        ideal=math.fsum(1/math.log2(k+2) for k in range(min(10,len(t))))
        actual=math.fsum(1/math.log2(k+2) for k,item in enumerate(ranking) if item in t)
        assert math.isclose(row['ndcg_at_10'],actual/ideal if ideal else 0,abs_tol=1e-14)
    reference=evaluate(np.tile(-np.arange(20),100),groups,poolmaps)[0]
    assert math.isclose(reference['ndcg_at_10'],shared['cbf']['ndcg_at_10'],abs_tol=1e-14)
    assert metric['users']==100 and metric['nonempty_truth_users']==11
    result={'status':'completed_lambdamart_june','same_100_users_and_order':True,
            'model_sha256':digest(model_path.read_bytes()),'no_retraining_or_june_tuning':True,
            'primary_lambdamart':metric,'cbf20_reference':reference,
            'shared_baselines':shared,'feature_and_predict_seconds':elapsed,
            'independent_per_user_ndcg_check':True,'llm_status':'awaiting_full_GPU_run',
            'files_sha256':{f.name:digest(f.read_bytes()) for f in out.iterdir() if f.is_file()}}
    (ROOT/'hw2/JUNE_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'evaluation_groups.json').write_text(json.dumps(groups))
    (out/'evaluation_mappings.json').write_text(json.dumps(poolmaps))
    example_index=next(i for i,g in enumerate(groups) if g['full_truth_count'])
    example={'user_alias':payloads[example_index]['user_alias'],
             'selection':'first nonempty-truth user in frozen order; illustration only',
             'truth_isins':sorted(truth[mappings[example_index]['user_id']]),
             'lambdamart_top10':[r['ISIN'] for r in preds[example_index*10:(example_index+1)*10]],
             'metrics':rows[example_index],'daria_personal_check':'pending'}
    (out/'review_example.json').write_text(json.dumps(example,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
