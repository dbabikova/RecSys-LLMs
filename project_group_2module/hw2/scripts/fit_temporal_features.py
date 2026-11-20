"""Fit encoders on train, transform validation; do not load June evaluation inputs."""
import gzip
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy
from scipy import sparse
import sklearn

from input_contract import canonical, digest
from ranker_features import RankerFeatures

ROOT = Path(__file__).resolve().parents[2]


def load_role(role):
    cfg = json.loads((ROOT/'hw2/TEMPORAL_CONFIG.json').read_text())
    payloads, groups = [], []
    for cut in cfg[role + '_cutoffs']:
        for suffix, dest in [('payloads',payloads),('labels',groups)]:
            path = ROOT/'data/hw2/temporal'/f'{role}_{cut}_{suffix}.jsonl.gz'
            dest.extend(json.loads(s) for s in gzip.decompress(path.read_bytes()).decode().splitlines())
    if [p['user_alias'] for p in payloads] != [g['group_key'] for g in groups]:
        raise ValueError('Payload/label group ordering mismatch')
    return payloads, groups


def main():
    cfg = json.loads((ROOT/'hw2/TEMPORAL_CONFIG.json').read_text())
    train, train_groups = load_role('train')
    validation, validation_groups = load_role('validation')
    encoder = RankerFeatures(cfg['text_settings']).fit(train)
    fit_hash = digest(canonical(encoder.fitted_state()))
    out = ROOT/'data/hw2/temporal/features'
    out.mkdir(exist_ok=True)
    report = {'fit_role':'train','fit_cutoffs':encoder.fit_cutoffs,'fit_group_count':len(train),
              'fit_names_count':len(encoder.fit_names), 'text_vocabulary_size':len(encoder.names.vocabulary_),
              'structured_feature_count':len(encoder.dictionary.feature_names_), 'total_features':encoder.feature_count,
              'fit_state_hash':fit_hash,'june_evaluation_loaded':False,'ranker_trained':False,
              'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
                             'scikit_learn':sklearn.__version__,'joblib':joblib.__version__},'splits':{}}
    for role, payloads, groups in [('train',train,train_groups),('validation',validation,validation_groups)]:
        matrix = encoder.transform(payloads)
        labels = np.asarray([v for g in groups for v in g['labels']],dtype=np.int8)
        group_sizes = np.asarray([len(g['labels']) for g in groups],dtype=np.int32)
        if len(labels) != matrix.shape[0] or sum(group_sizes) != len(labels):
            raise ValueError('Matrix/group/label shape mismatch')
        sparse.save_npz(out/f'{role}_X.npz',matrix)
        np.save(out/f'{role}_y.npy',labels,allow_pickle=False)
        np.save(out/f'{role}_group_sizes.npy',group_sizes,allow_pickle=False)
        report['splits'][role]={'groups':len(groups),'rows':matrix.shape[0],'columns':matrix.shape[1],
                              'nonzero_entries':matrix.nnz,'positive_labels':int(labels.sum()),
                              'zero_positive_groups':sum(not any(g['labels']) for g in groups),
                              **encoder.last_transform_diagnostics}
        print(json.dumps({role:report['splits'][role]}),flush=True)
    after = digest(canonical(encoder.fitted_state()))
    if after != fit_hash:
        raise ValueError('Validation mutated fitted state')
    joblib.dump(encoder,out/'train_encoder.joblib')
    (out/'feature_names.json').write_text(json.dumps(encoder.feature_names(),ensure_ascii=False)+'\n',encoding='utf-8')
    report['fit_state_unchanged_after_validation']=True
    report['artifacts_sha256']={p.name:digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
    report['code_sha256']={n:digest((Path(__file__).parent/n).read_bytes()) for n in ['fit_temporal_features.py','ranker_features.py']}
    (ROOT/'hw2/TEMPORAL_FEATURES.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
