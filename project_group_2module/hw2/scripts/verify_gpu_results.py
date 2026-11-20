"""Audit all raw GPU requests and independently score saved rankings."""
import csv
import json
import math
import shutil
import zipfile
from pathlib import Path
from input_contract import digest
from llm_ranker import build_prompt

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=Path('C:/Users/User/Downloads/ranker_results.zip')
    private=ROOT/'data/hw2/june_evaluation'
    shutil.copy2(source,private/'ranker_results.zip')
    shutil.copy2(Path('C:/Users/User/Downloads/Daria_HW2_GPU (1).ipynb'),private/'executed_notebook.ipynb')
    payloads=[json.loads(s) for s in (ROOT/'data/hw2/common_inputs/payloads.jsonl').read_text(encoding='utf-8').splitlines()]
    mappings=[json.loads(s) for s in (ROOT/'data/hw2/common_inputs/private_mapping.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=json.loads((private/'evaluation_groups.json').read_text())
    expected=json.loads((ROOT/'hw2/GPU_BUNDLE.json').read_text())['files_sha256']
    allrows=[]; predictions=[]; calls=[]; changed=0; original_identical=0
    with zipfile.ZipFile(source) as z:
        complete=json.loads(z.read('results/complete.json'))
        assert complete['users']==100 and complete['calls']==600
        assert complete['bundle_manifest']['files_sha256']==expected
        assert len(z.namelist())==204 and not any('interrupted' in n for n in z.namelist())
        runtime=json.loads(z.read('results/runtime.json'))
        pilot=json.loads(z.read('results/pilot_complete.json'))
        for p,m,g in zip(payloads,mappings,groups):
            uid=p['user_alias']; pool=[c['asset_alias'] for c in p['candidates']]
            assert uid==m['user_alias']==g['group_key']
            mapping=m['asset_alias_to_isin']
            saved=json.loads(z.read('results/'+uid+'.json'))
            logs=[json.loads(s) for s in z.read('results/'+uid+'_calls.jsonl').decode().splitlines()]
            assert len(logs)==6
            present={c['asset_alias'] for c in p['history']+p['candidates']}
            wire={a:f'A{i:02d}' for i,a in enumerate(sorted(present),1)}
            inverse={v:k for k,v in wire.items()}
            allowed={wire[a] for a in pool}
            for r,(branch,num,seed) in zip(logs,[('original',1,None),('original',2,None),('original',3,None),('permuted',1,17),('permuted',2,42),('permuted',3,2026)]):
                assert (r['user_alias'],r['branch'],r['call'],r['permutation_seed'])==(uid,branch,num,seed)
                assert r['prompt']==build_prompt(p,seed)
                assert r['prompt_sha256']==digest(r['prompt'].encode())
                assert r['status']=='completed_call' and r['generated_response_cache_hit'] is False
                assert 0<r['output_tokens']<=256 and r['input_tokens']+256<=16384
                try:
                    value=json.loads(r['text'].strip())
                    valid=(type(value) is list and len(value)==10 and all(type(a)is str for a in value)
                           and len(set(value))==10 and set(value)<=allowed)
                except (ValueError,TypeError): valid=False
                ranking=[inverse[a] for a in value] if valid else pool[:10]
                assert r['invalid']==r['fallback']==(not valid)
                assert r['ranked_aliases']==ranking
                calls.append(r)
            original_identical+=len({r['text'] for r in logs[:3]})==1
            for branch, subset in [('original',logs[:3]),('permuted',logs[3:])]:
                points={a:0 for a in pool}
                for r in subset:
                    for i,a in enumerate(r['ranked_aliases']): points[a]+=10-i
                ranking=sorted(pool,key=lambda a:(-points[a],mapping[a]))[:10]
                assert ranking==saved['branches'][branch]['aggregate_aliases']
                assert [r['ranked_aliases'] for r in subset]==saved['branches'][branch]['call_rankings']
                truth={a for a,label in zip(g['candidate_aliases'],g['labels']) if label}
                idcg=math.fsum(1/math.log2(i+2) for i in range(min(10,g['full_truth_count'])))
                dcg=math.fsum(1/math.log2(i+2) for i,a in enumerate(ranking) if a in truth)
                allrows.append({'user_alias':uid,'branch':branch,'truth_count':g['full_truth_count'],
                     'ndcg_at_10':dcg/idcg if idcg else 0.0,'fallback_calls':sum(r['fallback'] for r in subset)})
                for i,a in enumerate(ranking,1): predictions.append({'user_alias':uid,'branch':branch,'rank':i,'ISIN':mapping[a]})
            changed+=saved['branches']['original']['aggregate_aliases']!=saved['branches']['permuted']['aggregate_aliases']
    assert len(calls)==600
    metrics={}
    for branch in ['original','permuted']:
        rows=[r for r in allrows if r['branch']==branch]; active=[r for r in rows if r['truth_count']]
        logs=[r for r in calls if r['branch']==branch]
        metrics[branch]={'users':len(rows),'nonempty_truth_users':len(active),
             'ndcg_at_10':math.fsum(r['ndcg_at_10'] for r in rows)/100,
             'ndcg_at_10_nonempty':math.fsum(r['ndcg_at_10'] for r in active)/11,
             'calls':len(logs),'invalid_calls':sum(r['invalid'] for r in logs),
             'users_with_fallback':sum(r['fallback_calls']>0 for r in rows),
             'generation_seconds':math.fsum(r['seconds'] for r in logs),
             'input_tokens':sum(r['input_tokens'] for r in logs),'output_tokens':sum(r['output_tokens'] for r in logs)}
    deltas=[allrows[i+1]['ndcg_at_10']-allrows[i]['ndcg_at_10'] for i in range(0,200,2)]
    result={'status':'passed','users':100,'june_calls':600,'pilot_calls':6,'runtime':runtime,'pilot':pilot,
        'checks':{k:True for k in ['exact_input_bundle_hashes','all_600_frozen_prompts','strict_parser_and_fixed_fallback',
        'independent_aggregation_original_isin_ties','all_top10_unique_and_in_pool','no_cache_or_interrupted_calls',
        'full_truth_idcg_including_missing_assets','exact_100_users_and_order']},
        'metrics':metrics,'primary_delta_ndcg_at_10':math.fsum(deltas)/100,
        'paired_users_improved':sum(d>1e-14 for d in deltas),'paired_users_worsened':sum(d< -1e-14 for d in deltas),
        'paired_users_unchanged':sum(abs(d)<=1e-14 for d in deltas),
        'users_with_changed_aggregate':changed,'users_with_identical_original_raw_responses':original_identical,
        'archive_sha256':digest(source.read_bytes()),'executed_notebook_sha256':digest((private/'executed_notebook.ipynb').read_bytes()),
        'quality_tuning_after_june':False,'personal_raw_verification_by_daria':'pending'}
    for filename,rows in [('llm_per_user.csv',allrows),('llm_top10.csv',predictions)]:
        with (private/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (ROOT/'hw2/STAGE7_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    june_path=ROOT/'hw2/JUNE_RESULTS.json'
    june=json.loads(june_path.read_text())
    june.update(llm_status='completed_and_verified_600_calls',llm_metrics=metrics,
                llm_primary_delta=result['primary_delta_ndcg_at_10'],
                llm_results_archive_sha256=result['archive_sha256'])
    june_path.write_text(json.dumps(june,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
