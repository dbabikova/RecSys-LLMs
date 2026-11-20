"""Read-only source audit + baseline rerun; stdlib only, no model inference."""
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    project = Path(__file__).resolve().parents[2]
    stage = project / 'hw2'
    inputs = project / 'data/hw2/inputs'
    private = project / 'data/hw2/stage1_verification'
    private.mkdir(parents=True, exist_ok=True)
    sources = json.loads((stage / 'SOURCES.json').read_text(encoding='utf-8'))
    checks = {}
    for name, expected in sources['local_input_sha256'].items():
        require(sha((inputs / name).read_bytes()) == expected, f'Input hash mismatch: {name}')
    checks['local_input_hashes'] = True

    with zipfile.ZipFile(inputs / 'team5_alina_t4_protocol_check.zip') as package:
        require(package.testzip() is None, 'Corrupt baseline archive')
        files = {name: package.read(name) for name in package.namelist()}
    for name, expected in json.loads(files['SHA256.json']).items():
        require(sha(files[name]) == expected, f'Package member hash mismatch: {name}')
    require({name: sha(raw) for name, raw in files.items()} == sources['baseline_package_members_sha256'], 'Baseline package contents changed')
    checks['package_member_hashes'] = True
    protocol = json.loads(files['protocol.json'])
    digest = protocol.pop('protocol_hash')
    require(sha(json.dumps(protocol, sort_keys=True).encode()) == digest, 'Invalid protocol hash')
    protocol['protocol_hash'] = digest
    require(digest == sources['protocol_hash'], 'Unexpected protocol')
    checks['protocol_hash'] = True
    require(len(protocol['users']) == len(set(protocol['users'])) == 100, 'Wrong users')
    require(len(protocol['eligible_assets']) == 767, 'Wrong catalog')
    require(protocol['cutoff'] == '2021-06-01' and protocol['end_exclusive'] == '2021-07-01', 'Wrong dates')
    require(protocol['cohort_selection'] == 'warm' and protocol['exclude_seen'] is False, 'Wrong cohort or repeat policy')
    require(protocol['candidate_count'] == 50 and protocol['k'] == 10, 'Wrong source candidate settings')
    checks['teaching_settings'] = True
    require(files['commit.txt'].decode().strip() == sources['alina_experiment_commit'], 'Wrong source commit')
    runtime = json.loads(files['runtime.json'])
    require(runtime == sources['alina_runtime'], 'Wrong runtime attribution')
    checks['source_commit_and_runtime_attribution'] = True
    starter = inputs / 'starter_used_in_colab.py'
    teacher = inputs / 'teacher_08d4e734.py'
    require(starter.read_text(encoding='utf-8').rstrip() == teacher.read_text(encoding='utf-8').rstrip(), 'Teacher code differs')
    checks['teacher_code_equal_ignoring_trailing_whitespace'] = True

    data = private / 'data'
    data.mkdir(exist_ok=True)
    with zipfile.ZipFile(inputs / 'FAR-Trans.zip') as dataset:
        for name, expected in protocol['source_sha256'].items():
            raw = dataset.read('FAR-Trans/' + name)
            require(sha(raw) == expected, f'CSV hash mismatch: {name}')
            (data / name).write_bytes(raw)
    checks['source_csv_hashes'] = True
    out = private / 'baseline_rerun'
    command = [sys.executable, '-B', str(starter), 'run', '--data-dir', str(data), '--output-dir', str(out)]
    run = subprocess.run(command, capture_output=True, text=True)
    require(run.returncode == 0, 'Starter failed: ' + run.stderr)
    for name in ('protocol.json', 'metrics.json'):
        require(json.loads(files[name]) == json.loads((out / name).read_text(encoding='utf-8')), f'Rerun differs: {name}')
    checks['rerun_protocol_and_metrics'] = True
    contexts = [json.loads(s) for s in files['contexts.jsonl'].decode().splitlines()]
    require(contexts == [json.loads(s) for s in (out / 'contexts.jsonl').read_text(encoding='utf-8').splitlines()], 'Contexts differ')
    require([c['user_id'] for c in contexts] == protocol['users'], 'Context users/order differ')
    for c in contexts:
        ids = [a['id'] for a in c['candidate_assets']]
        require(len(ids) == len(set(ids)) == 50, 'Invalid source pool')
        require(set(ids) <= set(c['eligible_asset_ids']), 'Ineligible source candidate')
    checks['rerun_contexts_and_cbf50'] = True
    tables = {}
    for name in ('predictions.csv', 'per_user.csv'):
        tables[name] = list(csv.DictReader(io.StringIO(files[name].decode())))
        with (out / name).open(encoding='utf-8', newline='') as stream:
            require(tables[name] == list(csv.DictReader(stream)), f'Rerun differs: {name}')
    checks['rerun_predictions_and_per_user'] = True

    # Reconstruct labels independently, without importing starter functions.
    truth = defaultdict(set)
    seen = set()
    users = set(protocol['users'])
    eligible = set(protocol['eligible_assets'])
    with (data / 'transactions.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row.items())
            if key in seen:
                continue
            seen.add(key)
            if int(row['transactionID']) < 0:
                continue
            if row['customerID'] in users and row['transactionType'] == 'Buy' and protocol['cutoff'] <= row['timestamp'][:10] < protocol['end_exclusive'] and row['ISIN'] in eligible:
                truth[row['customerID']].add(row['ISIN'])
    methods = {'popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'}
    expected_pairs = {(u, m) for u in users for m in methods}
    predictions = defaultdict(list)
    for row in tables['predictions.csv']:
        predictions[row['user_id'], row['method']].append((int(row['rank']), row['ISIN']))
    require(set(predictions) == expected_pairs, 'Incomplete prediction coverage')
    scores = defaultdict(list)
    observed_pairs = set()
    for row in tables['per_user.csv']:
        pair = (row['user_id'], row['method'])
        require(pair in expected_pairs and pair not in observed_pairs, 'Wrong or duplicate per-user row')
        observed_pairs.add(pair)
        ranked_rows = sorted(predictions[pair])
        require([r for r, _ in ranked_rows] == list(range(1, 11)), 'Invalid ranks')
        ranked = [i for _, i in ranked_rows]
        require(len(set(ranked)) == 10 and set(ranked) <= eligible, 'Invalid Top-10')
        t = truth[pair[0]]
        dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(ranked) if i in t)
        ideal = sum(1 / math.log2(r + 2) for r in range(min(10, len(t))))
        score = dcg / ideal if ideal else 0.0
        require(int(row['relevant_count']) == len(t), 'Wrong truth count')
        require(math.isclose(float(row['ndcg_at_10']), score, abs_tol=1e-14), 'Wrong nDCG')
        scores[pair[1]].append((bool(t), score))
    require(observed_pairs == expected_pairs, 'Incomplete per-user coverage')
    metrics = json.loads(files['metrics.json'])['methods']
    for method, values in scores.items():
        nonempty = [s for present, s in values if present]
        require(len(values) == 100 and len(nonempty) == 11, 'Wrong denominator')
        require(math.isclose(sum(s for _, s in values) / 100, metrics[method]['ndcg_at_10'], abs_tol=1e-14), 'Wrong all-user mean')
        require(math.isclose(sum(nonempty) / 11, metrics[method]['ndcg_at_10_nonempty'], abs_tol=1e-14), 'Wrong nonempty mean')
    checks['independent_truth_top10_and_ndcg_400_rows'] = True
    checks['independent_aggregate_means'] = True
    report = {
        'status': 'passed', 'performed_by': 'Codex automated check; not Daria personal verification',
        'protocol_hash': digest, 'checks': checks, 'users': 100,
        'nonempty_truth_users': sum(bool(truth[u]) for u in users),
        'eligible_assets': 767, 'source_candidate_count': 50, 'baseline_rows_checked': 400,
        'model_training_or_inference_performed': False,
        'daria_review': 'pending', 'team_adoption': 'not established by this audit',
        'environment': {'python': sys.version.split()[0], 'platform': sys.platform},
        'verifier_sha256': sha(Path(__file__).read_bytes()),
    }
    (stage / 'VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
