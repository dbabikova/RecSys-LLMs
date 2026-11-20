"""Replay every saved model; independently reconstruct May truth and all nDCGs."""
import csv
import gzip
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import sparse

from input_contract import digest

ROOT=Path(__file__).resolve().parents[2]


def require(value,message):
    if not value:
        raise ValueError(message)


def main():
    result=json.loads((ROOT/'hw2/LAMBDAMART_RESULTS.json').read_text())
    cfg=json.loads((ROOT/'hw2/LAMBDAMART_CONFIG.json').read_text())
    require(result['config_sha256']==digest((ROOT/'hw2/LAMBDAMART_CONFIG.json').read_bytes()),'Config changed')
    require(result['source_feature_manifest_sha256']==digest((ROOT/'hw2/TEMPORAL_FEATURES.json').read_bytes()),'Feature manifest changed')
    require(result['june_evaluation_run'] is False and result['retrained_on_validation'] is False,'Unexpected evaluation/retraining')
    source=json.loads((ROOT/'hw2/SOURCES.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/FAR-Trans.zip') as z:
        raw={n:z.read('FAR-Trans/'+n) for n in source['source_csv_sha256']}
    for n,b in raw.items():
        require(digest(b)==source['source_csv_sha256'][n],'Raw data changed')
    known={r['ISIN'] for r in csv.DictReader(io.StringIO(raw['asset_information.csv'].decode('utf-8-sig'))) if r['timestamp'][:10]<'2021-05-01'}
    priced={r['ISIN'] for r in csv.DictReader(io.StringIO(raw['close_prices.csv'].decode('utf-8-sig'))) if r['timestamp'][:10]<'2021-05-01' and float(r['closePrice'])>0}
    eligible=known&priced
    truth=defaultdict(set)
    seen=set()
    for r in csv.DictReader(io.StringIO(raw['transactions.csv'].decode('utf-8-sig'))):
        sig=tuple(r.values())
        if sig in seen:
            continue
        seen.add(sig)
        if int(r['transactionID'])>=0 and r['transactionType']=='Buy' and '2021-05-01'<=r['timestamp'][:10]<'2021-05-31' and r['ISIN'] in eligible:
            truth[r['customerID']].add(r['ISIN'])
    base=ROOT/'data/hw2/temporal'
    mappings=[json.loads(s) for s in gzip.decompress((base/'validation_2021-05-01_mapping.jsonl.gz').read_bytes()).decode().splitlines()]
    groups=[json.loads(s) for s in gzip.decompress((base/'validation_2021-05-01_labels.jsonl.gz').read_bytes()).decode().splitlines()]
    require(len(mappings)==1000,'Wrong May cohort')
    x=sparse.load_npz(base/'features/validation_X.npz')
    out=ROOT/'data/hw2/lambdamart'
    original_scores=[]
    for m,g in zip(mappings,groups):
        t=truth[m['user_id']]
        require(m['truth_isins']==sorted(t) and g['full_truth_count']==len(t),'May labels mismatch')
        idcg=sum(1/math.log2(i+2) for i in range(min(10,len(t))))
        dcg=sum(1/math.log2(i+2) for i,isin in enumerate(m['pool_isins'][:10]) if isin in t)
        original_scores.append(dcg/idcg if idcg else 0.0)
    reference_mean=sum(original_scores)/1000
    require(math.isclose(reference_mean,result['validation_cbf20_reference']['ndcg_at_10'],rel_tol=0,abs_tol=1e-14),'Reference metric mismatch')
    all_scores={}
    for run in result['runs']:
        directory=out/f'{run["configuration"]}_seed{run["seed"]}'
        require(digest((directory/'model.txt').read_bytes())==run['model_sha256'],'Model bytes changed')
        require(digest((directory/'validation_scores.npy').read_bytes())==run['validation_scores_sha256'],'Scores changed')
        model=lgb.Booster(model_file=str(directory/'model.txt'))
        require(model.num_feature()==x.shape[1] and model.num_trees()==run['actual_trees'],'Model shape mismatch')
        scores=model.predict(x,num_threads=4)
        saved=np.load(directory/'validation_scores.npy',allow_pickle=False)
        require(np.array_equal(scores,saved),'Saved model predictions do not reproduce exactly')
        require(np.isfinite(scores).all(),'Nonfinite model scores')
        with (directory/'validation_top10.csv').open(encoding='utf-8',newline='') as f:
            predictions=list(csv.DictReader(f))
        with (directory/'validation_per_user.csv').open(encoding='utf-8',newline='') as f:
            per_user=list(csv.DictReader(f))
        require(len(predictions)==10000 and len(per_user)==1000,'Incomplete validation outputs')
        values=[]
        for index,(m,g,r) in enumerate(zip(mappings,groups,per_user)):
            ids=m['pool_isins']
            order=sorted(range(20),key=lambda i:(-scores[index*20+i],ids[i]))[:10]
            actual=predictions[index*10:(index+1)*10]
            require(all(p['group_key']==g['group_key'] for p in actual),'Prediction cohort/order mismatch')
            require([int(p['rank']) for p in actual]==list(range(1,11)),'Invalid ranks')
            require([p['ISIN'] for p in actual]==[ids[i] for i in order],'Top-10 differs from score/ISIN order')
            require([p['candidate_alias'] for p in actual]==[g['candidate_aliases'][i] for i in order],'Alias mapping mismatch')
            t=truth[m['user_id']]
            dcg=sum(1/math.log2(rank+2) for rank,i in enumerate(order) if ids[i] in t)
            idcg=sum(1/math.log2(rank+2) for rank in range(min(10,len(t))))
            value=dcg/idcg if idcg else 0.0
            require(r['group_key']==g['group_key'] and int(r['full_truth_count'])==len(t),'Wrong per-user denominator')
            require(math.isclose(float(r['ndcg_at_10']),value,rel_tol=0,abs_tol=1e-14),'Wrong per-user nDCG')
            values.append(value)
        active=[v for v,m in zip(values,mappings) if truth[m['user_id']]]
        require(len(active)==171,'Nonempty population changed')
        metric=run['metrics']
        require(math.isclose(sum(values)/1000,metric['ndcg_at_10'],rel_tol=0,abs_tol=1e-14),'Wrong all-user mean')
        require(math.isclose(sum(active)/171,metric['ndcg_at_10_nonempty'],rel_tol=0,abs_tol=1e-14),'Wrong nonempty mean')
        all_scores[run['configuration'],run['seed']]=values
    grid={c['id']:c for c in cfg['grid']}
    selection=[r for r in result['runs'] if r['phase']=='configuration_selection']
    require(len(selection)==4 and all(r['seed']==0 for r in selection),'Wrong selection budget')
    best=min(selection,key=lambda r:(-r['metrics']['ndcg_at_10'],grid[r['configuration']]['num_boost_round'],grid[r['configuration']]['num_leaves'],r['configuration']))
    require(result['primary_configuration']==best['configuration'] and result['primary_seed']==0 and result['primary_model_sha256']==best['model_sha256'],'Selection rule violated')
    require(sorted(r['seed'] for r in result['runs'] if r['configuration']==best['configuration'])==[0,1,2,3,4],'Wrong seed coverage')
    values=all_scores[best['configuration'],0]
    report={'status':'passed','performed_by':'Codex automated verification; Daria review pending',
            'models_replayed':len(result['runs']),'per_user_ndcg_checked':len(result['runs'])*1000,
            'top10_rows_checked':len(result['runs'])*10000,'validation_users':1000,'nonempty_truth_users':171,
            'primary_model_sha256':best['model_sha256'],
            'paired_primary_vs_reference':{'improved':sum(a>b+1e-14 for a,b in zip(values,original_scores)),
                                           'declined':sum(a<b-1e-14 for a,b in zip(values,original_scores)),
                                           'unchanged':sum(abs(a-b)<=1e-14 for a,b in zip(values,original_scores)),
                                           'delta_all':best['metrics']['ndcg_at_10']-reference_mean},
            'checks':{k:True for k in ['raw_may_truth_and_full_idcg','all_users_and_zero_truth_retained',
                     'valid_top10_and_isin_tie_break','all_per_user_and_aggregate_metrics',
                     'saved_model_predictions_bitwise_reproduced','configuration_selection_rule','fixed_primary_seed_and_five_seed_coverage',
                     'source_model_and_score_hashes']},'june_evaluation_run':False}
    (ROOT/'hw2/STAGE5_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
