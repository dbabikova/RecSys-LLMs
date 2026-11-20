"""Independent raw-CSV input audit. Imports neither builder nor contract nor starter."""
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(value):
    return '[MISSING]' if not value or value.strip().casefold() in ('', 'not_available', 'nan') else value.strip()


def age(cut, when):
    return (date.fromisoformat(cut) - date.fromisoformat(when[:10])).days


def earlier(value, cut):
    try:
        return value[:10] != '2000-01-01' and date.fromisoformat(value[:10]) < date.fromisoformat(cut)
    except (ValueError, TypeError):
        return False


def main():
    stage, private = ROOT / 'hw2', ROOT / 'data/hw2/common_inputs'
    manifest = json.loads((stage / 'INPUTS.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        require(sha((private / name).read_bytes()) == expected, f'File hash failed: {name}')
    expected_hash = manifest.pop('input_manifest_hash')
    require(sha(json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()) == expected_hash, 'Manifest hash failed')
    load = lambda n: [json.loads(s) for s in (private / n).read_text(encoding='utf-8').splitlines()]
    payloads, features, mappings, orders = (load(n) for n in ['payloads.jsonl', 'feature_records.jsonl', 'private_mapping.jsonl', 'presentation_orders.jsonl'])
    with zipfile.ZipFile(ROOT / 'data/hw2/inputs/team5_alina_t4_protocol_check.zip') as z:
        protocol = json.loads(z.read('protocol.json'))
    pools = [json.loads(s) for s in (ROOT / 'data/hw2/cbf20/candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    users, cut = protocol['users'], protocol['cutoff']
    require([m['user_id'] for m in mappings] == users, 'Cohort/order changed')
    require(len(payloads) == len(orders) == 100 and len(features) == 2000, 'Incomplete coverage')
    assets, profiles, counts, latest = {}, {}, defaultdict(Counter), defaultdict(dict)
    with zipfile.ZipFile(ROOT / 'data/hw2/inputs/FAR-Trans.zip') as z:
        for name, key, target in [('asset_information.csv', 'ISIN', assets), ('customer_information.csv', 'customerID', profiles)]:
            raw = z.read('FAR-Trans/' + name)
            require(sha(raw) == protocol['source_sha256'][name], 'Raw hash failed')
            for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
                if r['timestamp'][:10] < cut and (r[key] not in target or r['timestamp'] > target[r[key]]['timestamp']):
                    target[r[key]] = r
        raw = z.read('FAR-Trans/transactions.csv')
        require(sha(raw) == protocol['source_sha256']['transactions.csv'], 'Transactions hash failed')
        seen = set()
        for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            signature = tuple(r.values())
            if signature in seen:
                continue
            seen.add(signature)
            if int(r['transactionID']) < 0 or r['timestamp'][:10] >= cut or r['transactionType'] != 'Buy':
                continue
            u, i = r['customerID'], r['ISIN']
            if u in users and i in protocol['eligible_assets']:
                counts[u][i] += 1
                latest[u][i] = max(latest[u].get(i, ''), r['timestamp'][:10])
    assets = {i: assets[i] for i in protocol['eligible_assets']}
    vectors = {}
    for i, r in assets.items():
        ts = sorted({f'{f}={r[f].strip().casefold()}' for f in FIELDS if normalize(r.get(f)) != '[MISSING]'})
        vectors[i] = {t: 1 / math.sqrt(len(ts)) for t in ts}
    by_pair = {(r['user_alias'], r['candidate_alias']): r for r in features}
    require(len(by_pair) == 2000, 'Duplicate feature pairs')
    withheld = truncated = 0
    for pos, (payload, mapping, pool, order) in enumerate(zip(payloads, mappings, pools, orders), 1):
        uid = users[pos - 1]
        require(payload['user_alias'] == mapping['user_alias'] == order['user_alias'] == f'U{pos:03d}', 'Alias order mismatch')
        require(payload['cutoff'] == cut and payload['repeat_purchases_allowed'] is True, 'Wrong settings')
        reverse = mapping['asset_alias_to_isin']
        require(all(a == 'A' + sha(i.encode())[:12] for a, i in reverse.items()), 'Alias mapping changed')
        require([reverse[c['asset_alias']] for c in payload['candidates']] == pool['candidate_ids'], 'Candidate identity/order changed')
        retained = sorted(counts[uid], key=lambda i: (-date.fromisoformat(latest[uid][i]).toordinal(), -counts[uid][i], i))[:20]
        require([reverse[h['asset_alias']] for h in payload['history']] == retained, 'History truncation/order mismatch')
        summary = {'full_distinct_assets': len(counts[uid]), 'full_buy_events': sum(counts[uid].values()),
                   'retained_distinct_assets': len(retained), 'retained_buy_events': sum(counts[uid][i] for i in retained),
                   'omitted_distinct_assets': len(counts[uid]) - len(retained)}
        require(payload['history_summary'] == summary, 'History summary mismatch')
        truncated += bool(summary['omitted_distinct_assets'])
        row = profiles[uid]
        snapshot_ok = earlier(row['timestamp'], cut)
        q_ok = snapshot_ok and earlier(row['lastQuestionnaireDate'], cut)
        require(payload['profile']['snapshot_age_days'] == (age(cut, row['timestamp']) if snapshot_ok else None), 'Snapshot age mismatch')
        for f in ['customerType', 'riskLevel', 'investmentCapacity']:
            allowed = snapshot_ok and (f == 'customerType' or q_ok)
            v = normalize(row[f])
            status = ('withheld_unverified' if not allowed else 'missing' if v == '[MISSING]' else 'estimated' if v.startswith('Predicted_') else 'observed')
            require(payload['profile'][f] == {'value': v if allowed else '[MISSING]', 'status': status}, 'Profile masking mismatch')
        withheld += not q_ok
        for item in payload['history'] + payload['candidates']:
            isin = reverse[item['asset_alias']]
            require(item['name'] == normalize(assets[isin]['assetName']), 'Asset name mismatch')
            require(item['attributes'] == {f: normalize(assets[isin][f]) for f in FIELDS}, 'Asset fields mismatch')
            require(item['metadata_age_days'] == age(cut, assets[isin]['timestamp']) > 0, 'Metadata age mismatch')
            require(item['last_buy_age_days'] == (age(cut, latest[uid][isin]) if isin in latest[uid] else None), 'Buy recency mismatch')
            count_field = 'buy_count' if item in payload['history'] else 'past_buy_count'
            require(item[count_field] == counts[uid].get(isin, 0), 'Buy count mismatch')
        profile_vector = defaultdict(float)
        for isin, n in counts[uid].items():
            for t, v in vectors[isin].items():
                profile_vector[t] += n * v
        norm = math.sqrt(sum(v*v for v in profile_vector.values()))
        for rank, c in enumerate(payload['candidates'], 1):
            isin = reverse[c['asset_alias']]
            expected_score = sum(profile_vector[t]*v for t,v in vectors[isin].items()) / norm if norm else 0.0
            require(c['source_cbf_rank'] == rank and math.isclose(c['source_cbf_score'], expected_score, rel_tol=0, abs_tol=1e-12), 'CBF score/rank mismatch')
            f = by_pair[payload['user_alias'], c['asset_alias']]
            require(f['history_context'] == payload['history'], 'LambdaMART history differs from LLM')
            require(f['text'] == {'candidate_name': c['name'], 'history_names': [h['name'] for h in payload['history']]}, 'Text information omitted')
            expected_numeric = {k: c[k] for k in ['metadata_age_days','source_cbf_rank','source_cbf_score','past_buy_count','last_buy_age_days']}
            expected_numeric['cutoff_ordinal'] = date.fromisoformat(cut).toordinal()
            expected_numeric.update(summary)
            expected_numeric['profile_snapshot_age_days'] = payload['profile']['snapshot_age_days']
            cat = dict(c['attributes'])
            for key in ['customerType','riskLevel','investmentCapacity']:
                cat[key] = payload['profile'][key]['value']
                cat[key + '_status'] = payload['profile'][key]['status']
            require(f['categorical'] == cat, 'Categorical information differs')
            for key in FIELDS:
                total = sum(h['buy_count'] for h in payload['history'])
                matching = sum(h['buy_count'] for h in payload['history'] if c['attributes'][key] != '[MISSING]' and h['attributes'][key] == c['attributes'][key])
                expected_numeric[key + '_history_buy_share'] = matching / total if total else 0.0
            require(f['numeric'] == expected_numeric, 'Numeric features differ')
        require(order['original'] == [c['asset_alias'] for c in payload['candidates']], 'Original order mismatch')
        for seed, ids in order['permutations'].items():
            expected = sorted(order['original'], key=lambda a: (sha(f'{seed}:{a}'.encode()), a))
            require(ids == expected and set(ids) == set(order['original']), 'Presentation permutation changed membership')
    for name in ['payloads.jsonl','feature_records.jsonl']:
        text = (private / name).read_text(encoding='utf-8')
        require(not any(u in text for u in users), 'Raw customerID exposed')
        # Strict record schemas and field-by-field raw checks above constrain semantics.
        require(not any('"' + k + '"' in text for k in ['truth','relevant_count','ndcg_at_10','oracle','totalValue','units','transactionID','lastQuestionnaireDate']), 'Forbidden field in model inputs')
    report = {'status':'passed', 'performed_by':'Codex automated verification; Daria review pending',
              'users':100, 'feature_records_checked':2000, 'candidate_hash':manifest['source_candidate_hash'],
              'input_manifest_hash':expected_hash, 'history_truncated_users':truncated, 'withheld_risk_capacity_users':withheld,
              'checks':{k:True for k in ['exact_source_cohort_and_cbf20','raw_metadata_and_buy_counts','raw_recency_and_truncation',
                        'questionnaire_masking','full_history_cbf_scores','llm_and_unfitted_feature_information',
                        'permutation_ids_and_order','no_customer_ids_or_label_fields','file_and_manifest_hashes']},
              'model_training_or_inference_performed':False, 'text_transform_fitted':False}
    (stage / 'STAGE3_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
