"""Independent audit of frozen candidates/reference and post-hoc pool diagnostics.

Does not import the builder or starter. Labels stay in ignored diagnostics files.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def mean(values):
    return math.fsum(values) / len(values) if values else 0.0


def ndcg(ranking, truth):
    ideal = math.fsum(1 / math.log2(r + 2) for r in range(min(10, len(truth))))
    actual = math.fsum(1 / math.log2(r + 2) for r, isin in enumerate(ranking[:10]) if isin in truth)
    return actual / ideal if ideal else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-dir', type=Path, default=PROJECT / 'data/hw2/cbf20')
    parser.add_argument('--report', type=Path, default=PROJECT / 'hw2/STAGE2_VERIFICATION.json')
    args = parser.parse_args()
    sources = json.loads((PROJECT / 'hw2/SOURCES.json').read_text(encoding='utf-8'))
    stage_sources = json.loads((PROJECT / 'hw2/STAGE2_SOURCES.json').read_text(encoding='utf-8'))
    inputs = PROJECT / 'data/hw2/inputs'
    package = inputs / 'team5_alina_t4_protocol_check.zip'
    require(sha(package.read_bytes()) == sources['local_input_sha256'][package.name], 'Changed source ZIP')
    with zipfile.ZipFile(package) as z:
        raw = {n: z.read(n) for n in ['protocol.json', 'contexts.jsonl', 'predictions.csv', 'per_user.csv', 'metrics.json']}
    for n, b in raw.items():
        require(sha(b) == sources['baseline_package_members_sha256'][n], f'Source hash mismatch: {n}')
    protocol = json.loads(raw['protocol.json'])
    # Uliana's derivative export preserves the run manifest and protocol loader.
    # Matching protocol FILE bytes bind exact user IDs/order, not merely the count.
    uliana_file = inputs / 'uliana_tselishcheva_hw8_session.json'
    require(sha(uliana_file.read_bytes()) == stage_sources['uliana_session_sha256'], 'Uliana export changed')
    uliana = json.loads(uliana_file.read_text(encoding='utf-8-sig'))
    artifacts = {a['name']: a for a in uliana['artifacts']}
    uliana_manifest = json.loads(artifacts['outputs/experiment/run_manifest.json']['text'])
    require(uliana_manifest['inputs']['baseline_members_sha256']['protocol.json'] == sha(raw['protocol.json']), 'Uliana protocol bytes differ')
    require(uliana_manifest['inputs']['protocol_hash'] == protocol['protocol_hash'], 'Uliana protocol hash differs')
    require(uliana_manifest['inputs']['users'] == 100, 'Uliana cohort count differs')
    require(uliana_manifest['inputs']['alina_zip_sha256'] == sha(package.read_bytes()), 'Uliana source ZIP differs')
    source_contexts = [json.loads(s) for s in raw['contexts.jsonl'].decode().splitlines()]
    records = [json.loads(s) for s in (args.artifact_dir / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((args.artifact_dir / 'candidate_manifest.json').read_text(encoding='utf-8'))
    users = protocol['users']
    require(len(users) == len(set(users)) == 100, 'Wrong source cohort')
    require([r['user_id'] for r in records] == users == [c['user_id'] for c in source_contexts], 'Different user IDs/order')
    pools = {}
    for r, c in zip(records, source_contexts):
        ids = r['candidate_ids']
        require(len(ids) == len(set(ids)) == 20, 'Pool size/uniqueness failed')
        require(ids == [a['id'] for a in c['candidate_assets'][:20]], 'Pool is not the exact CBF50 prefix')
        require(r['candidate_assets'] == c['candidate_assets'][:20], 'Candidate metadata changed')
        require(set(ids) <= set(c['eligible_asset_ids']) <= set(protocol['eligible_assets']), 'Invalid eligibility')
        require(set(r) == {'user_id', 'candidate_ids', 'candidate_assets'}, 'Unexpected candidate record fields')
        pools[r['user_id']] = ids
    expected_hash = sha(encoded([{'user_id': r['user_id'], 'candidate_ids': r['candidate_ids']} for r in records]))
    require(manifest['candidate_hash'] == expected_hash, 'Candidate hash failed')
    require(manifest['cohort_hash'] == sha(encoded(users)), 'Cohort hash failed')
    require(manifest['eligible_catalog_hash'] == sha(encoded(protocol['eligible_assets'])), 'Catalog hash failed')
    require(manifest['source_protocol_hash'] == protocol['protocol_hash'], 'Wrong source protocol hash')
    identity = {k: manifest[k] for k in ['source_protocol_hash', 'policy', 'candidate_hash', 'cohort_hash', 'eligible_catalog_hash', 'source_contexts_sha256']}
    require(sha(encoded(identity)) == manifest['candidate_manifest_hash'], 'Manifest identity hash failed')
    require(manifest['policy'] == {'name': 'saved_cbf50_first20_v1', 'source_count': 50, 'take': 20, 'preserve_user_order': True, 'preserve_candidate_order': True, 'reference_k': 10, 'uses_future_labels': False}, 'Unexpected policy')
    for name, expected in manifest['files_sha256'].items():
        require(sha((args.artifact_dir / name).read_bytes()) == expected, f'Frozen file hash failed: {name}')
    require(manifest == json.loads((PROJECT / 'hw2/CANDIDATES.json').read_text(encoding='utf-8')), 'Public/private manifest mismatch')
    references = list(csv.DictReader((args.artifact_dir / 'reference_predictions.csv').open(encoding='utf-8', newline='')))
    expected_rows = [{'method': 'candidate_order_first10', 'user_id': u, 'rank': str(i + 1), 'ISIN': isin} for u in users for i, isin in enumerate(pools[u][:10])]
    require(references == expected_rows, 'Reference rows/order mismatch')
    cbf = defaultdict(list)
    for r in csv.DictReader(io.StringIO(raw['predictions.csv'].decode())):
        if r['method'] == 'cbf':
            cbf[r['user_id']].append((int(r['rank']), r['ISIN']))
    require(set(cbf) == set(users), 'Source CBF cohort mismatch')
    require(all([isin for _, isin in sorted(cbf[u])] == pools[u][:10] for u in users), 'Reference differs from baseline CBF')

    # Labels are reconstructed ONLY after fixed candidate/reference checks above.
    dataset = inputs / 'FAR-Trans.zip'
    require(sha(dataset.read_bytes()) == sources['local_input_sha256'][dataset.name], 'Dataset ZIP changed')
    with zipfile.ZipFile(dataset) as z:
        transactions = z.read('FAR-Trans/transactions.csv')
    require(sha(transactions) == sources['source_csv_sha256']['transactions.csv'], 'Transaction CSV changed')
    truth, history = defaultdict(set), defaultdict(set)
    seen = set()
    eligible, user_set = set(protocol['eligible_assets']), set(users)
    for r in csv.DictReader(io.StringIO(transactions.decode('utf-8-sig'))):
        key = tuple(r.items())
        if key in seen:
            continue
        seen.add(key)
        if int(r['transactionID']) < 0 or r['customerID'] not in user_set or r['transactionType'] != 'Buy':
            continue
        uid, isin, date = r['customerID'], r['ISIN'], r['timestamp'][:10]
        if date < protocol['cutoff']:
            history[uid].add(isin)
        elif date < protocol['end_exclusive'] and isin in eligible:
            truth[uid].add(isin)
    saved = {r['user_id']: r for r in csv.DictReader(io.StringIO(raw['per_user.csv'].decode())) if r['method'] == 'cbf'}
    values = []
    for index, uid in enumerate(users, 1):
        t, pool = truth[uid], pools[uid]
        hits = set(pool) & t
        repeat = t & history[uid]
        new = t - history[uid]
        ref = ndcg(pool[:10], t)
        # Ideal arrangement inside pool, divided by full-truth IDCG (not pool-truth IDCG).
        idcg = math.fsum(1 / math.log2(r + 2) for r in range(min(10, len(t))))
        best_dcg = math.fsum(1 / math.log2(r + 2) for r in range(min(10, len(hits))))
        oracle = best_dcg / idcg if idcg else 0.0
        require(int(saved[uid]['relevant_count']) == len(t), 'Truth count differs from source')
        require(math.isclose(float(saved[uid]['ndcg_at_10']), ref, abs_tol=1e-14), 'Reference nDCG differs from source')
        require(ref <= oracle + 1e-14 <= 1 + 1e-14, 'Invalid oracle bound')
        values.append({'user_id': uid, 'cohort_position_1based': index, 'truth_count': len(t),
                       'pool_hits': len(hits), 'recall20': len(hits) / len(t) if t else None,
                       'reference_ndcg10': ref, 'oracle_ndcg10': oracle,
                       'repeat_truth_count': len(repeat), 'repeat_pool_hits': len(repeat & set(pool)),
                       'new_truth_count': len(new), 'new_pool_hits': len(new & set(pool))})
    active = [v for v in values if v['truth_count']]
    require(len(active) == 11, 'Wrong nonempty cohort')
    m = json.loads(raw['metrics.json'])['methods']['cbf']
    require(math.isclose(mean([v['reference_ndcg10'] for v in values]), m['ndcg_at_10'], abs_tol=1e-14), 'All-user reference mean failed')
    require(math.isclose(mean([v['reference_ndcg10'] for v in active]), m['ndcg_at_10_nonempty'], abs_tol=1e-14), 'Nonempty reference mean failed')
    diagnostics = {'reference_ndcg10_all': mean([v['reference_ndcg10'] for v in values]),
                   'reference_ndcg10_nonempty': mean([v['reference_ndcg10'] for v in active]),
                   'oracle_ndcg10_all': mean([v['oracle_ndcg10'] for v in values]),
                   'oracle_ndcg10_nonempty': mean([v['oracle_ndcg10'] for v in active]),
                   'candidate_recall20_macro_nonempty': mean([v['recall20'] for v in active]),
                   'candidate_recall20_micro': sum(v['pool_hits'] for v in active) / sum(v['truth_count'] for v in active),
                   'future_distinct_user_asset_pairs': sum(v['truth_count'] for v in values),
                   'future_pairs_in_pool': sum(v['pool_hits'] for v in values),
                   'nonempty_users_with_any_pool_hit': sum(v['pool_hits'] > 0 for v in active),
                   'nonempty_users_without_pool_hits': sum(v['pool_hits'] == 0 for v in active),
                   'nonempty_users_with_all_truth_in_pool': sum(v['pool_hits'] == v['truth_count'] for v in active)}
    for kind in ['repeat', 'new']:
        total = sum(v[f'{kind}_truth_count'] for v in values)
        found = sum(v[f'{kind}_pool_hits'] for v in values)
        diagnostics[kind + '_pairs'] = total
        diagnostics[kind + '_pairs_in_pool'] = found
        diagnostics[kind + '_recall_micro'] = found / total if total else None
    private = PROJECT / 'data/hw2/stage2_diagnostics'
    private.mkdir(exist_ok=True)
    with (private / 'per_user.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(values)
    # Deterministic review example, selected for illustration only; no cohort/pool changes.
    chosen = active[0]
    uid = chosen['user_id']
    example = {**chosen, 'selection_rule': 'first user in frozen cohort order with nonempty truth; diagnostics only',
               'candidate_ids': pools[uid], 'truth_ids': sorted(truth[uid]),
               'truth_positions_in_pool': {isin: pools[uid].index(isin) + 1 if isin in pools[uid] else None for isin in sorted(truth[uid])},
               'personal_verification': 'pending; automated expected evidence only'}
    (private / 'review_example.json').write_text(json.dumps(example, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = {'status': 'passed', 'performed_by': 'Codex automated check; Daria review pending',
              'source_protocol_hash': protocol['protocol_hash'], 'candidate_hash': expected_hash,
              'candidate_manifest_hash': manifest['candidate_manifest_hash'],
              'checks': {k: True for k in ['exact_same_100_user_ids_and_order', 'uliana_manifest_same_protocol_bytes_and_source_zip', 'exact_cbf50_first20_ids_and_metadata',
                         'unique_eligible_candidates', 'manifest_and_file_hashes', 'reference_1000_rows',
                         'reference_equals_source_cbf', 'independent_truth_and_ndcg_100_users', 'aggregate_means_and_oracle_bounds']},
              'users': 100, 'nonempty_truth_users': 11, 'eligible_assets': 767,
              'candidate_count': 20, 'diagnostics': diagnostics,
              'diagnostic_policy': 'Post-hoc labels used for audit only. No tuning, selection, or model inputs.',
              'recall_denominators': {'macro': '11 nonempty users; empty-truth recall undefined and excluded',
                                      'micro': 'all distinct eligible future user-asset pairs'},
              'oracle_denominator': 'full truth IDCG@10, including relevant assets missing from pool',
              'verifier_sha256': sha(Path(__file__).read_bytes()),
              'model_training_or_inference_performed': False}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
