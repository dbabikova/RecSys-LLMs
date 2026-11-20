"""Full-truth nDCG for fixed candidate scores; empty truth contributes zero."""
import math
import numpy as np


def evaluate(scores, groups, mappings):
    scores = np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or len(scores) != sum(len(g['labels']) for g in groups):
        raise ValueError('Invalid prediction scores')
    if len(groups) != len(mappings):
        raise ValueError('Group/mapping count mismatch')
    rows, predictions, offset = [],[],0
    for group, mapping in zip(groups,mappings):
        if group['group_key'] != mapping['group_key'] or len(group['labels']) != len(mapping['pool_isins']):
            raise ValueError('Group/mapping alignment mismatch')
        n=len(group['labels'])
        ids=mapping['pool_isins']
        if len(set(ids))!=n:
            raise ValueError('Duplicate candidates')
        order=sorted(range(n),key=lambda i:(-scores[offset+i],ids[i]))[:10]
        truth_count=group['full_truth_count']
        dcg=math.fsum(group['labels'][i]/math.log2(rank+2) for rank,i in enumerate(order))
        idcg=math.fsum(1/math.log2(rank+2) for rank in range(min(10,truth_count)))
        value=dcg/idcg if idcg else 0.0
        rows.append({'group_key':group['group_key'],'full_truth_count':truth_count,
                     'positive_candidate_count':sum(group['labels']),'ndcg_at_10':value})
        for rank,i in enumerate(order,1):
            predictions.append({'group_key':group['group_key'],'rank':rank,'ISIN':ids[i],
                                'candidate_alias':group['candidate_aliases'][i]})
        offset+=n
    active=[r['ndcg_at_10'] for r in rows if r['full_truth_count']]
    metric={'users':len(rows),'nonempty_truth_users':len(active),'empty_truth_users':len(rows)-len(active),
            'ndcg_at_10':math.fsum(r['ndcg_at_10'] for r in rows)/len(rows),
            'ndcg_at_10_nonempty':math.fsum(active)/len(active) if active else 0.0,'failed_cases':0}
    return metric,rows,predictions
