"""Fixed-grid train/May selection and five seeds; never load June evaluation data."""
import csv
import gzip
import json
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import sparse

from input_contract import canonical,digest
from ranker_metrics import evaluate

ROOT=Path(__file__).resolve().parents[2]


def load_may(suffix):
    path=ROOT/'data/hw2/temporal'/f'validation_2021-05-01_{suffix}.jsonl.gz'
    return [json.loads(s) for s in gzip.decompress(path.read_bytes()).decode().splitlines()]


def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    started=time.perf_counter()
    cfg_path=ROOT/'hw2/LAMBDAMART_CONFIG.json'
    cfg=json.loads(cfg_path.read_text())
    if lgb.__version__!=cfg['lightgbm_version']:
        raise ValueError('Unexpected LightGBM version')
    out=ROOT/'data/hw2/lambdamart'
    out.mkdir(exist_ok=True)
    if (out/'run_complete.json').exists():
        raise ValueError('Completed run exists; preserve results rather than overwrite')
    feature_dir=ROOT/'data/hw2/temporal/features'
    feature_manifest=json.loads((ROOT/'hw2/TEMPORAL_FEATURES.json').read_text())
    for name,expected in feature_manifest['artifacts_sha256'].items():
        if digest((feature_dir/name).read_bytes())!=expected:
            raise ValueError('Training feature source changed')
    x=sparse.load_npz(feature_dir/'train_X.npz')
    y=np.load(feature_dir/'train_y.npy',allow_pickle=False)
    sizes=np.load(feature_dir/'train_group_sizes.npy',allow_pickle=False)
    valid=sparse.load_npz(feature_dir/'validation_X.npz')
    groups,mappings=load_may('labels'),load_may('mapping')
    dataset=lgb.Dataset(x,label=y,group=sizes,free_raw_data=False)
    load_seconds=time.perf_counter()-started
    baseline_scores=np.tile(-np.arange(20,dtype=float),len(groups))
    reference=evaluate(baseline_scores,groups,mappings)[0]
    records=[]
    def fit(config,seed,phase):
        directory=out/f'{config["id"]}_seed{seed}'
        directory.mkdir(exist_ok=True)
        params=dict(cfg['base_parameters'],objective=cfg['objective'],num_leaves=config['num_leaves'],
                    seed=seed,feature_fraction_seed=seed,bagging_seed=seed)
        before=time.perf_counter()
        model=lgb.train(params,dataset,num_boost_round=config['num_boost_round'])
        fit_seconds=time.perf_counter()-before
        before=time.perf_counter()
        scores=model.predict(valid,num_threads=4)
        predict_seconds=time.perf_counter()-before
        metric,rows,predictions=evaluate(scores,groups,mappings)
        before=time.perf_counter()
        model.save_model(str(directory/'model.txt'))
        np.save(directory/'validation_scores.npy',scores,allow_pickle=False)
        save_csv(directory/'validation_per_user.csv',rows)
        save_csv(directory/'validation_top10.csv',predictions)
        record={'configuration':config['id'],'seed':seed,'phase':phase,'parameters':params,
                'requested_trees':config['num_boost_round'],'actual_trees':model.num_trees(),
                'metrics':metric,'fit_seconds':fit_seconds,'validation_predict_seconds':predict_seconds,
                'save_seconds':time.perf_counter()-before,
                'model_sha256':digest((directory/'model.txt').read_bytes()),
                'validation_scores_sha256':digest((directory/'validation_scores.npy').read_bytes())}
        (directory/'run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        records.append(record)
        print(json.dumps({k:record[k] for k in ['configuration','seed','phase','metrics','fit_seconds']}),flush=True)
        return record
    for config in cfg['grid']:
        fit(config,cfg['selection_seed'],'configuration_selection')
    config_by_id={c['id']:c for c in cfg['grid']}
    best=min(records,key=lambda r:(-r['metrics']['ndcg_at_10'],config_by_id[r['configuration']]['num_boost_round'],
                                   config_by_id[r['configuration']]['num_leaves'],r['configuration']))
    chosen=config_by_id[best['configuration']]
    for seed in cfg['diagnostic_seeds']:
        if seed!=cfg['selection_seed']:
            fit(chosen,seed,'seed_diagnostic')
    seeds=[r for r in records if r['configuration']==chosen['id']]
    values=[r['metrics']['ndcg_at_10'] for r in seeds]
    result={'status':'complete','config_sha256':digest(cfg_path.read_bytes()),
            'source_feature_manifest_sha256':digest((ROOT/'hw2/TEMPORAL_FEATURES.json').read_bytes()),
            'primary_configuration':chosen['id'],'primary_seed':0,'primary_model_sha256':best['model_sha256'],
            'primary_validation_metrics':best['metrics'],'validation_cbf20_reference':reference,
            'selection_rule':cfg['selection_rule'],'seed_selection':'fixed seed 0; other seeds diagnostic only',
            'seed_summary':{'seeds':[r['seed'] for r in seeds],'all_user_mean':float(np.mean(values)),
                            'all_user_sd_ddof1':float(np.std(values,ddof=1)),
                            'all_user_min':min(values),'all_user_max':max(values)},
            'runs':records,'runtime':{'load_and_setup_seconds':load_seconds,'whole_run_seconds':time.perf_counter()-started,
                                    'scope':'CPU training grid + seed runs, May inference, scoring and writes; no June or LLM inference'},
            'environment':{'python':sys.version.split()[0],'lightgbm':lgb.__version__,
                           'platform':platform.system()+' '+platform.release(),'machine':platform.machine(),
                           'logical_cpu_count':os.cpu_count(),'training_threads':4,
                           'venv':'project-local, system-site-packages inherited; LightGBM installed only here'},
            'june_evaluation_run':False,'retrained_on_validation':False,
            'code_sha256':{n:digest((Path(__file__).parent/n).read_bytes()) for n in ['train_lambdamart.py','ranker_metrics.py']}}
    (out/'run_complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (ROOT/'hw2/LAMBDAMART_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
