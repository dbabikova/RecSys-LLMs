"""Freeze the first 20 saved CBF candidates. No truth or raw transactions read."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
POLICY = {'name': 'saved_cbf50_first20_v1', 'source_count': 50, 'take': 20,
          'preserve_user_order': True, 'preserve_candidate_order': True,
          'reference_k': 10, 'uses_future_labels': False}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze_file(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Refusing to overwrite different frozen artifact: {path.name}')
    if not path.exists():
        path.write_bytes(raw)


def main():
    sources = json.loads((PROJECT / 'hw2/SOURCES.json').read_text(encoding='utf-8'))
    archive = PROJECT / 'data/hw2/inputs/team5_alina_t4_protocol_check.zip'
    if sha(archive.read_bytes()) != sources['local_input_sha256'][archive.name]:
        raise ValueError('Baseline ZIP hash mismatch')
    with zipfile.ZipFile(archive) as z:
        # Only these two members are read. Evaluation truth is not a builder input.
        raw_protocol, raw_contexts = z.read('protocol.json'), z.read('contexts.jsonl')
    for name, raw in [('protocol.json', raw_protocol), ('contexts.jsonl', raw_contexts)]:
        if sha(raw) != sources['baseline_package_members_sha256'][name]:
            raise ValueError(f'Source member changed: {name}')
    protocol = json.loads(raw_protocol)
    contexts = [json.loads(s) for s in raw_contexts.decode().splitlines()]
    if [c['user_id'] for c in contexts] != protocol['users'] or len(set(protocol['users'])) != 100:
        raise ValueError('Source cohort/order mismatch')
    records = []
    for c in contexts:
        candidates = c['candidate_assets']
        ids = [a['id'] for a in candidates]
        if len(ids) != len(set(ids)) or len(ids) != 50:
            raise ValueError('Invalid CBF50')
        if not set(ids) <= set(c['eligible_asset_ids']) <= set(protocol['eligible_assets']):
            raise ValueError('Ineligible source candidate')
        records.append({'user_id': c['user_id'], 'candidate_ids': ids[:20],
                        'candidate_assets': candidates[:20]})
    ordered_ids = [{'user_id': r['user_id'], 'candidate_ids': r['candidate_ids']} for r in records]
    candidate_hash = sha(canonical(ordered_ids))
    identity = {'source_protocol_hash': protocol['protocol_hash'], 'policy': POLICY,
                'candidate_hash': candidate_hash,
                'cohort_hash': sha(canonical(protocol['users'])),
                'eligible_catalog_hash': sha(canonical(protocol['eligible_assets'])),
                'source_contexts_sha256': sha(raw_contexts)}
    candidate_bytes = b''.join(canonical(r) + b'\n' for r in records)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['method', 'user_id', 'rank', 'ISIN'])
    for r in records:
        for rank, isin in enumerate(r['candidate_ids'][:10], 1):
            writer.writerow(['candidate_order_first10', r['user_id'], rank, isin])
    reference_bytes = stream.getvalue().encode('utf-8')
    manifest = {**identity, 'candidate_manifest_hash': sha(canonical(identity)),
                'hash_encoding': 'SHA-256 of UTF-8 JSON: sort_keys=True, ensure_ascii=False, separators=(comma, colon); ordered lists unchanged',
                'builder_sha256': sha(Path(__file__).read_bytes()),
                'users': 100, 'candidate_count': 20,
                'files_sha256': {'candidates.jsonl': sha(candidate_bytes),
                                 'reference_predictions.csv': sha(reference_bytes)},
                'scope': 'Candidate selection only; feature contract and model settings pending stage 3+'}
    out = PROJECT / 'data/hw2/cbf20'
    out.mkdir(parents=True, exist_ok=True)
    freeze_file(out / 'candidates.jsonl', candidate_bytes)
    freeze_file(out / 'reference_predictions.csv', reference_bytes)
    freeze_file(out / 'candidate_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b'\n')
    # Public manifest contains hashes and aggregate counts, no user IDs or asset lists.
    (PROJECT / 'hw2/CANDIDATES.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'users': len(records), 'candidate_count': 20,
                      'candidate_hash': candidate_hash,
                      'candidate_manifest_hash': manifest['candidate_manifest_hash']}, indent=2))


if __name__ == '__main__':
    main()
