"""Build common payloads from frozen cohort/candidates and strictly past source rows."""
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from input_contract import (FIELDS, HISTORY_LIMIT, alias, canonical, common_payload,
                            digest, feature_records, presentation)

PROJECT = Path(__file__).resolve().parents[2]


def write_frozen(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Different frozen stage-3 file: {path.name}')
    if not path.exists():
        path.write_bytes(raw)


def main():
    sources = json.loads((PROJECT / 'hw2/SOURCES.json').read_text(encoding='utf-8'))
    cm = json.loads((PROJECT / 'hw2/CANDIDATES.json').read_text(encoding='utf-8'))
    inputs = PROJECT / 'data/hw2/inputs'
    archive = inputs / 'FAR-Trans.zip'
    if digest(archive.read_bytes()) != sources['local_input_sha256']['FAR-Trans.zip']:
        raise ValueError('Wrong dataset')
    with zipfile.ZipFile(inputs / 'team5_alina_t4_protocol_check.zip') as z:
        raw_protocol, raw_contexts = z.read('protocol.json'), z.read('contexts.jsonl')
    for name, raw in [('protocol.json', raw_protocol), ('contexts.jsonl', raw_contexts)]:
        if digest(raw) != sources['baseline_package_members_sha256'][name]:
            raise ValueError('Wrong baseline source')
    protocol = json.loads(raw_protocol)
    contexts = [json.loads(s) for s in raw_contexts.decode().splitlines()]
    private_pool = PROJECT / 'data/hw2/cbf20/candidates.jsonl'
    if digest(private_pool.read_bytes()) != cm['files_sha256']['candidates.jsonl']:
        raise ValueError('Frozen pool changed')
    pools = [json.loads(s) for s in private_pool.read_text(encoding='utf-8').splitlines()]
    if [p['user_id'] for p in pools] != protocol['users'] or [c['user_id'] for c in contexts] != protocol['users']:
        raise ValueError('Cohort/order changed')
    cutoff, users, eligible = protocol['cutoff'], set(protocol['users']), set(protocol['eligible_assets'])
    assets, profiles = {}, {}
    counts, latest = defaultdict(Counter), defaultdict(dict)
    with zipfile.ZipFile(archive) as z:
        for name, key, target in [('asset_information.csv', 'ISIN', assets), ('customer_information.csv', 'customerID', profiles)]:
            raw = z.read('FAR-Trans/' + name)
            if digest(raw) != sources['source_csv_sha256'][name]:
                raise ValueError('Source CSV changed')
            for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
                if row['timestamp'][:10] < cutoff and (row[key] not in target or row['timestamp'] > target[row[key]]['timestamp']):
                    target[row[key]] = row
        assets = {i: assets[i] for i in protocol['eligible_assets']}
        raw = z.read('FAR-Trans/transactions.csv')
        if digest(raw) != sources['source_csv_sha256']['transactions.csv']:
            raise ValueError('Transactions changed')
        seen = set()
        for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            # Do not construct any future labels; discard future rows before aggregation.
            if row['timestamp'][:10] >= cutoff:
                continue
            key = tuple(sorted(row.items()))
            if key in seen:
                continue
            seen.add(key)
            uid, isin = row['customerID'], row['ISIN']
            if int(row['transactionID']) < 0 or uid not in users or isin not in eligible or row['transactionType'] != 'Buy':
                continue
            counts[uid][isin] += 1
            latest[uid][isin] = max(latest[uid].get(isin, ''), row['timestamp'][:10])
    if len({alias(i) for i in assets}) != len(assets):
        raise ValueError('Asset alias collision')
    payloads, features, mappings, summary = [], [], [], []
    for index, (context, pool) in enumerate(zip(contexts, pools), 1):
        uid, ua = pool['user_id'], f'U{index:03d}'
        if dict(counts[uid]) != {h['id']: h['buy_count'] for h in context['history']}:
            raise ValueError('Source history counts differ')
        payload = common_payload(ua, cutoff, assets, profiles[uid], counts[uid], latest[uid], pool['candidate_ids'])
        payloads.append(payload)
        features.extend(feature_records(payload))
        mappings.append({'user_alias': ua, 'user_id': uid,
                         'asset_alias_to_isin': {alias(i): i for i in set(pool['candidate_ids']) | set(counts[uid])}})
        summary.append({'history_truncated': payload['history_summary']['omitted_distinct_assets'] > 0,
                        'risk_status': payload['profile']['riskLevel']['status'],
                        'capacity_status': payload['profile']['investmentCapacity']['status']})
    out = PROJECT / 'data/hw2/common_inputs'
    out.mkdir(exist_ok=True)
    files = {'payloads.jsonl': payloads, 'feature_records.jsonl': features, 'private_mapping.jsonl': mappings}
    hashes = {}
    for name, records in files.items():
        raw = b''.join(canonical(r) + b'\n' for r in records)
        write_frozen(out / name, raw)
        hashes[name] = digest(raw)
    presentations = [{'user_alias': p['user_alias'], 'original': [c['asset_alias'] for c in p['candidates']],
                      'permutations': {str(s): [c['asset_alias'] for c in presentation(p, s)['candidates']] for s in [17, 42, 2026]}}
                     for p in payloads]
    raw = b''.join(canonical(r) + b'\n' for r in presentations)
    write_frozen(out / 'presentation_orders.jsonl', raw)
    hashes['presentation_orders.jsonl'] = digest(raw)
    contract = {'version': 'common_input_v1', 'cutoff': cutoff, 'users': len(payloads), 'candidate_count': 20,
                'history_limit_distinct_assets': HISTORY_LIMIT,
                'history_selection': 'last Buy descending, buy_count descending, ISIN ascending',
                'profile_rule': 'risk/capacity withheld if questionnaire or profile date missing/sentinel/not strictly before cut',
                'alias_rule': 'asset A + first 12 hex SHA256(ISIN); user U + cohort position',
                'source_candidate_hash': cm['candidate_hash'], 'source_protocol_hash': protocol['protocol_hash'],
                'source_csv_sha256': sources['source_csv_sha256'], 'files_sha256': hashes,
                'feature_records': len(features), 'fitted_text_or_categorical_encoder': False,
                'history_truncated_users': sum(s['history_truncated'] for s in summary),
                'risk_status_counts': dict(sorted(Counter(s['risk_status'] for s in summary).items())),
                'capacity_status_counts': dict(sorted(Counter(s['capacity_status'] for s in summary).items())),
                'code_sha256': {n: digest((Path(__file__).parent / n).read_bytes()) for n in ['build_inputs.py', 'input_contract.py']}}
    contract['input_manifest_hash'] = digest(canonical(contract))
    (PROJECT / 'hw2/INPUTS.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
