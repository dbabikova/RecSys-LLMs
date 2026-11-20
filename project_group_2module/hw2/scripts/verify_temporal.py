"""Independent temporal/cohort/label/CBF audit plus train-only encoder checks."""
import csv
import gzip
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from input_contract import canonical, digest
import ranker_features  # Resolve our locally produced joblib class; no external pickle input.

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('assetCategory','assetSubCategory','sector','industry','marketID')


def require(value, message):
    if not value:
        raise ValueError(message)


def load(name):
    return [json.loads(s) for s in gzip.decompress((ROOT/'data/hw2/temporal'/name).read_bytes()).decode().splitlines()]


def main():
    cfg = json.loads((ROOT/'hw2/TEMPORAL_CONFIG.json').read_text())
    report = json.loads((ROOT/'hw2/TEMPORAL_DATA.json').read_text())
    fit_report = json.loads((ROOT/'hw2/TEMPORAL_FEATURES.json').read_text())
    require(report['config_sha256']==digest((ROOT/'hw2/TEMPORAL_CONFIG.json').read_bytes()),'Config changed')
    require(max((date.fromisoformat(c)+timedelta(days=30)).isoformat() for c in cfg['train_cutoffs']) <= min(cfg['validation_cutoffs']), 'Train labels cross validation cutoff')
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/FAR-Trans.zip') as z:
        raw = {n:z.read('FAR-Trans/'+n) for n in report['source_csv_sha256']}
    for n,b in raw.items():
        require(digest(b)==report['source_csv_sha256'][n],'Raw source hash mismatch')
    expected_groups = defaultdict(list)
    all_payloads = defaultdict(list)
    checked = positives = 0
    for item in report['cuts']:
        role,cut = item['role'],item['cutoff']
        end = (date.fromisoformat(cut)+timedelta(days=30)).isoformat()
        require(end==item['end_exclusive'] and end < '2021-06-01','Wrong temporal boundary')
        for n,h in item['files_sha256'].items():
            require(digest((ROOT/'data/hw2/temporal'/n).read_bytes())==h,'Temporal file hash mismatch')
        assets, profiles = {},{}
        for name,key,target in [('asset_information.csv','ISIN',assets),('customer_information.csv','customerID',profiles)]:
            for r in csv.DictReader(io.StringIO(raw[name].decode('utf-8-sig'))):
                if r['timestamp'][:10] < cut and (r[key] not in target or target[r[key]]['timestamp'] < r['timestamp']):
                    target[r[key]]=r
        priced={r['ISIN'] for r in csv.DictReader(io.StringIO(raw['close_prices.csv'].decode('utf-8-sig'))) if r['timestamp'][:10] < cut and float(r['closePrice'])>0}
        assets={i:a for i,a in assets.items() if i in priced}
        history, latest, truth = defaultdict(Counter),defaultdict(dict),defaultdict(set)
        seen=set()
        for r in csv.DictReader(io.StringIO(raw['transactions.csv'].decode('utf-8-sig'))):
            sig=tuple(r.values())
            if sig in seen:
                continue
            seen.add(sig)
            if int(r['transactionID'])<0 or r['transactionType']!='Buy' or r['ISIN'] not in assets:
                continue
            u,i,d=r['customerID'],r['ISIN'],r['timestamp'][:10]
            if d<cut:
                history[u][i]+=1
                latest[u][i]=max(latest[u].get(i,''),d)
            elif d<end:
                truth[u].add(i)
        users=sorted((u for u in profiles if history.get(u)),key=lambda u:(digest(('42:'+u).encode()),u))[:1000]
        require(digest(canonical(users))==item['ordered_users_hash'],'Wrong past-only user selection')
        prefix=f'{role}_{cut}'
        payloads,groups,mapping=(load(f'{prefix}_{s}.jsonl.gz') for s in ['payloads','labels','mapping'])
        require([m['user_id'] for m in mapping]==users,'User IDs/order changed')
        require(len(payloads)==len(groups)==len(mapping)==1000,'Incomplete temporal cohort')
        vec={}
        for i,a in assets.items():
            ts=sorted({f'{k}={a[k].strip().casefold()}' for k in FIELDS if a.get(k,'').strip() not in ('','Not_Available','nan')})
            vec[i]={t:1/math.sqrt(len(ts)) for t in ts}
        for idx,(p,g,m,u) in enumerate(zip(payloads,groups,mapping,users),1):
            key=f'{cut}:U{idx:04d}'
            require(p['user_alias']==g['group_key']==m['group_key']==key,'Group key/order mismatch')
            require(p['cutoff']==cut and g['role']==role and g['end_exclusive']==end,'Group temporal fields mismatch')
            profile=defaultdict(float)
            for i,n in history[u].items():
                for t,v in vec[i].items():
                    profile[t]+=n*v
            norm=math.sqrt(sum(v*v for v in profile.values()))
            scores={i:sum(profile[t]*v for t,v in vector.items())/norm if norm else 0 for i,vector in vec.items()}
            require(any(scores.values()),'Unanticipated CBF cold fallback; implement explicitly')
            pool=sorted(scores,key=lambda i:(-scores[i],i))[:20]
            aliases=['A'+digest(i.encode())[:12] for i in pool]
            require(m['pool_isins']==pool and g['candidate_aliases']==aliases==[c['asset_alias'] for c in p['candidates']],'CBF20 differs from independent reconstruction')
            require(g['labels']==[int(i in truth[u]) for i in pool],'Wrong labels')
            require(g['full_truth_count']==len(truth[u]) and m['truth_isins']==sorted(truth[u]),'Wrong full truth')
            retained=sorted(history[u],key=lambda i:(-date.fromisoformat(latest[u][i]).toordinal(),-history[u][i],i))[:20]
            require([h['asset_alias'] for h in p['history']]==['A'+digest(i.encode())[:12] for i in retained],'History cutoff/truncation mismatch')
            for h,i in zip(p['history'],retained):
                require(h['buy_count']==history[u][i] and h['last_buy_age_days']==(date.fromisoformat(cut)-date.fromisoformat(latest[u][i])).days>0,'Historical input mismatch')
            for rank,(c,i) in enumerate(zip(p['candidates'],pool),1):
                require(c['source_cbf_rank']==rank and math.isclose(c['source_cbf_score'],scores[i],abs_tol=1e-12),'Source score mismatch')
                require(c['past_buy_count']==history[u].get(i,0),'Candidate history mismatch')
                require(c['metadata_age_days']==(date.fromisoformat(cut)-date.fromisoformat(assets[i]['timestamp'][:10])).days>0,'Future metadata')
            require(not any(k in canonical(p).decode() for k in ['"truth"','"labels"','"full_truth_count"','"user_id"']),'Label or ID in payload')
            checked+=1
            positives+=sum(g['labels'])
        expected_groups[role].extend(groups)
        all_payloads[role].extend(payloads)
    feature_dir=ROOT/'data/hw2/temporal/features'
    for n,h in fit_report['artifacts_sha256'].items():
        require(digest((feature_dir/n).read_bytes())==h,'Feature artifact hash mismatch')
    encoder=joblib.load(feature_dir/'train_encoder.joblib')
    require(encoder.fit_group_keys==[g['group_key'] for g in expected_groups['train']],'Encoder fitted on wrong groups')
    names=sorted({i['name'] for p in all_payloads['train'] for i in p['history']+p['candidates']})
    require(encoder.fit_names==names,'Text corpus includes non-train names')
    analyzed=[set(encoder.names.build_analyzer()(s)) for s in names]
    for token,index in encoder.names.vocabulary_.items():
        df=sum(token in terms for terms in analyzed)
        expected_idf=math.log((1+len(names))/(1+df))+1
        require(math.isclose(encoder.names.idf_[index],expected_idf,abs_tol=1e-12),'IDF not fitted on train-only corpus')
    require(digest(canonical(encoder.fitted_state()))==fit_report['fit_state_hash'],'Fitted state hash mismatch')
    matrix_counts={}
    for role in ['train','validation']:
        x=sparse.load_npz(feature_dir/f'{role}_X.npz')
        y=np.load(feature_dir/f'{role}_y.npy',allow_pickle=False)
        sizes=np.load(feature_dir/f'{role}_group_sizes.npy',allow_pickle=False)
        groups=expected_groups[role]
        require(sizes.tolist()==[20]*len(groups) and sum(sizes)==len(y)==x.shape[0],'Matrix/group dimensions mismatch')
        require(y.tolist()==[v for g in groups for v in g['labels']],'Matrix label order mismatch')
        require(x.shape[1]==encoder.feature_count and np.isfinite(x.data).all(),'Invalid feature matrix')
        for index in sorted({0,len(groups)//2,len(groups)-1}):
            regenerated=encoder.transform([all_payloads[role][index]])
            require((regenerated-x[index*20:(index+1)*20]).nnz==0,'Saved matrix does not reproduce from serialized encoder')
        matrix_counts[role]={'groups':len(groups),'rows':x.shape[0],'positive_labels':int(y.sum())}
    result={'status':'passed','performed_by':'Codex automated check; Daria review pending',
            'groups_raw_verified':checked,'positive_candidate_labels':positives,'matrix_counts':matrix_counts,
            'checks':{k:True for k in ['nonoverlapping_train_validation_labels','all_labels_end_before_june',
                    'past_only_warm_cohorts','asof_eligible_universe_per_cut','independent_cbf20_and_binary_labels',
                    'history_recency_and_no_label_fields','train_only_fit_group_keys_and_text_corpus',
                    'independent_train_only_idf','saved_matrix_labels_group_order_and_finiteness',
                    'serialized_encoder_reproduces_fixed_groups','all_artifact_hashes']},
            'ranker_trained':False,'june_evaluation_inputs_used_for_fit':False}
    (ROOT/'hw2/STAGE4_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
