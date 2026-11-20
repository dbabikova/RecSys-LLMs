"""Build past-window ranker data using frozen contract and pinned starter retrieval."""
import gzip
import importlib.util
import json
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from input_contract import alias, canonical, common_payload, digest

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = json.loads((ROOT / 'hw2/TEMPORAL_CONFIG.json').read_text())
    sources = json.loads((ROOT / 'hw2/SOURCES.json').read_text(encoding='utf-8'))
    inputs = ROOT / 'data/hw2/inputs'
    starter_path = inputs / 'starter_used_in_colab.py'
    if digest(starter_path.read_bytes()) != sources['local_input_sha256'][starter_path.name]:
        raise ValueError('Starter changed')
    spec = importlib.util.spec_from_file_location('pinned_temporal_starter', starter_path)
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    out = ROOT / 'data/hw2/temporal'
    out.mkdir(exist_ok=True)
    csv_dir = out / 'raw'
    csv_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(inputs / 'FAR-Trans.zip') as z:
        for name, expected in sources['source_csv_sha256'].items():
            raw = z.read('FAR-Trans/' + name)
            if digest(raw) != expected:
                raise ValueError('CSV source changed')
            (csv_dir / name).write_bytes(raw)
    report = {'config_sha256':digest((ROOT / 'hw2/TEMPORAL_CONFIG.json').read_bytes()),
              'source_csv_sha256':sources['source_csv_sha256'], 'cuts':[],
              'june_evaluation_loaded':False, 'ranker_trained':False}
    for role in ['train', 'validation']:
        for cut in cfg[role + '_cutoffs']:
            end = (date.fromisoformat(cut) + timedelta(days=cfg['label_horizon_days'])).isoformat()
            if end >= cfg['evaluation_cutoff']:
                raise ValueError('Temporal labels reach evaluation cutoff')
            assets, profiles, history, truth, actual_end, excluded = starter.load_data(csv_dir, cut, cfg['label_horizon_days'])
            users = starter.select_users(profiles, history, 'warm', cfg['max_warm_users_per_cutoff'], None)
            tokens = {i:starter.tokens(a) for i,a in assets.items()}
            events = starter.scan_recent_events(csv_dir, cut, assets)
            latest = defaultdict(dict)
            for uid in users:
                for d, i, kind in events.get(uid, ()):
                    if kind == 'Buy':
                        latest[uid][i] = max(latest[uid].get(i,''),d)
            groups, payloads, mapping = [], [], []
            for index, uid in enumerate(users,1):
                source_scores = starter.content_scores(history[uid], tokens)
                pool = starter.order(source_scores)[:20]
                ua = f'{cut}:U{index:04d}'
                payload = common_payload(ua, cut, assets, profiles[uid], history[uid], latest[uid], pool)
                relevant = truth.get(uid,set()) & set(assets)
                group = {'group_key':ua, 'role':role, 'cutoff':cut, 'end_exclusive':end,
                         'candidate_aliases':[alias(i) for i in pool],
                         'labels':[int(i in relevant) for i in pool],
                         'full_truth_count':len(relevant), 'positive_candidates':len(set(pool)&relevant)}
                if len(set(group['candidate_aliases'])) != 20:
                    raise ValueError('Invalid temporal pool')
                payloads.append(payload)
                groups.append(group)
                mapping.append({'group_key':ua,'user_id':uid,'pool_isins':pool,'truth_isins':sorted(relevant)})
            prefix = f'{role}_{cut}'
            hashes = {}
            for suffix, records in [('payloads',payloads),('labels',groups),('mapping',mapping)]:
                raw = b''.join(canonical(r)+b'\n' for r in records)
                path = out / f'{prefix}_{suffix}.jsonl.gz'
                compressed = gzip.compress(raw,mtime=0)
                if path.exists() and path.read_bytes() != compressed:
                    raise ValueError('Refusing to overwrite different temporal data')
                path.write_bytes(compressed)
                hashes[path.name]=digest(compressed)
            nonempty = sum(g['full_truth_count']>0 for g in groups)
            with_positive = sum(g['positive_candidates']>0 for g in groups)
            item = {'role':role,'cutoff':cut,'end_exclusive':end,'eligible_assets':len(assets),
                    'users':len(users),'candidate_rows':len(users)*20,'nonempty_truth_users':nonempty,
                    'groups_with_positive_candidates':with_positive,
                    'groups_without_positive_candidates':len(groups)-with_positive,
                    'positive_candidate_labels':sum(g['positive_candidates'] for g in groups),
                    'ordered_users_hash':digest(canonical(users)), 'files_sha256':hashes}
            report['cuts'].append(item)
            print(json.dumps(item),flush=True)
    report['code_sha256']={n:digest((Path(__file__).parent/n).read_bytes()) for n in ['build_temporal.py','input_contract.py']}
    (ROOT/'hw2/TEMPORAL_DATA.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
