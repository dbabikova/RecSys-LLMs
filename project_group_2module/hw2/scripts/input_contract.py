"""Past-only common inputs and unfitted feature records; no model dependencies."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date

FIELDS = ('assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID')
HISTORY_LIMIT = 20
MISSING = '[MISSING]'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def clean(value):
    value = (value or '').strip()
    return MISSING if value.casefold() in ('', 'not_available', 'nan') else value


def alias(isin):
    return 'A' + digest(isin.encode())[:12]


def age(cutoff, earlier):
    return (date.fromisoformat(cutoff) - date.fromisoformat(earlier[:10])).days


def valid_earlier(value, cutoff):
    try:
        return value[:10] != '2000-01-01' and date.fromisoformat(value[:10]) < date.fromisoformat(cutoff)
    except (ValueError, TypeError):
        return False


def tokens(row):
    return {f'{f}={row[f].strip().casefold()}' for f in FIELDS if clean(row.get(f)) != MISSING}


def cbf_scores(counts, assets):
    vectors = {i: {t: 1 / math.sqrt(len(ts)) for t in ts}
               for i, row in assets.items() for ts in [sorted(tokens(row))]}
    profile = defaultdict(float)
    for i, count in counts.items():
        for t, value in vectors[i].items():
            profile[t] += count * value
    norm = math.sqrt(sum(v * v for v in profile.values()))
    return {i: sum(profile[t] * value for t, value in v.items()) / norm if norm else 0.0
            for i, v in vectors.items()}


def profile_input(row, cutoff):
    snapshot_ok = valid_earlier(row.get('timestamp', ''), cutoff)
    questionnaire_ok = snapshot_ok and valid_earlier(row.get('lastQuestionnaireDate', ''), cutoff)
    result = {'snapshot_age_days': age(cutoff, row['timestamp']) if snapshot_ok else None}
    for f in ('customerType', 'riskLevel', 'investmentCapacity'):
        value = clean(row.get(f))
        allowed = snapshot_ok and (f == 'customerType' or questionnaire_ok)
        result[f] = {'value': value if allowed else MISSING,
                     'status': ('withheld_unverified' if not allowed else
                                'missing' if value == MISSING else
                                'estimated' if value.startswith('Predicted_') else 'observed')}
    return result


def asset_input(isin, assets, cutoff):
    row = assets[isin]
    if row['timestamp'][:10] >= cutoff:
        raise ValueError('Future metadata')
    return {'asset_alias': alias(isin), 'name': clean(row.get('assetName')),
            'attributes': {f: clean(row.get(f)) for f in FIELDS},
            'metadata_age_days': age(cutoff, row['timestamp'])}


def common_payload(uid_alias, cutoff, assets, profile, counts, latest, pool):
    if any(i not in latest or latest[i] >= cutoff for i in counts):
        raise ValueError('Missing or future Buy date')
    scores = cbf_scores(counts, assets)
    retained = sorted(counts, key=lambda i: (-date.fromisoformat(latest[i]).toordinal(), -counts[i], i))[:HISTORY_LIMIT]
    history = []
    for isin in retained:
        history.append(dict(asset_input(isin, assets, cutoff), buy_count=counts[isin],
                            last_buy_age_days=age(cutoff, latest[isin])))
    candidates = []
    for rank, isin in enumerate(pool, 1):
        candidates.append(dict(asset_input(isin, assets, cutoff), source_cbf_rank=rank,
                               source_cbf_score=scores[isin], past_buy_count=counts.get(isin, 0),
                               last_buy_age_days=age(cutoff, latest[isin]) if isin in latest else None))
    return {'user_alias': uid_alias, 'cutoff': cutoff, 'repeat_purchases_allowed': True,
            'profile': profile_input(profile, cutoff),
            'history_summary': {'full_distinct_assets': len(counts), 'full_buy_events': sum(counts.values()),
                                'retained_distinct_assets': len(retained),
                                'retained_buy_events': sum(counts[i] for i in retained),
                                'omitted_distinct_assets': len(counts) - len(retained)},
            'history': history, 'candidates': candidates}


def feature_records(payload):
    """Flatten common input for a later train-only encoder, not a fitted ranker matrix."""
    profile, summary, history = payload['profile'], payload['history_summary'], payload['history']
    records = []
    for candidate in payload['candidates']:
        row = {'user_alias': payload['user_alias'], 'candidate_alias': candidate['asset_alias'],
               'numeric': {'cutoff_ordinal': date.fromisoformat(payload['cutoff']).toordinal(),
                           **{k: candidate[k] for k in ['metadata_age_days', 'source_cbf_rank', 'source_cbf_score', 'past_buy_count', 'last_buy_age_days']}},
               'categorical': dict(candidate['attributes']),
               'text': {'candidate_name': candidate['name'],
                        'history_names': [h['name'] for h in history]},
               'history_context': history}
        row['numeric'].update({'profile_snapshot_age_days': profile['snapshot_age_days'], **summary})
        for f in ('customerType', 'riskLevel', 'investmentCapacity'):
            row['categorical'][f] = profile[f]['value']
            row['categorical'][f + '_status'] = profile[f]['status']
        for f in FIELDS:
            weight = sum(h['buy_count'] for h in history)
            matching = sum(h['buy_count'] for h in history
                           if candidate['attributes'][f] != MISSING and h['attributes'][f] == candidate['attributes'][f])
            row['numeric'][f + '_history_buy_share'] = matching / weight if weight else 0.0
        records.append(row)
    return records


def presentation(payload, seed=None):
    """Permutation changes presentation only; source rank/score stay attached."""
    result = json.loads(canonical(payload))
    if seed is not None:
        result['candidates'].sort(key=lambda c: (digest(f'{seed}:{c["asset_alias"]}'.encode()), c['asset_alias']))
    return result
