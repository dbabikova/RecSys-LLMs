"""Train-fitted sparse features from the shared payload; no relevance labels used."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from input_contract import FIELDS, feature_records


def structured_rows(payload):
    for record in feature_records(payload):
        row = {}
        for k, value in record['numeric'].items():
            row['n:' + k] = -1.0 if value is None else float(value)
            row['missing:' + k] = float(value is None)
        row.update({'c:' + k:v for k,v in record['categorical'].items()})
        # Preserve each retained history slot's non-text fields, not only match shares.
        for slot, item in enumerate(payload['history']):
            for k in ['buy_count','last_buy_age_days','metadata_age_days']:
                row[f'h{slot}:{k}'] = float(item[k])
            for k in FIELDS:
                row[f'h{slot}:{k}'] = item['attributes'][k]
        yield row


class RankerFeatures:
    def __init__(self, text_settings):
        self.text_settings = dict(text_settings)
        self.dictionary = DictVectorizer(sparse=True, sort=True, dtype=np.float64)
        settings = dict(text_settings)
        settings['ngram_range'] = tuple(settings['ngram_range'])
        self.names = TfidfVectorizer(**settings, token_pattern=r'(?u)\b\w+\b', dtype=np.float64)

    def fit(self, train_payloads):
        if not train_payloads:
            raise ValueError('Empty training payloads')
        self.dictionary.fit(row for p in train_payloads for row in structured_rows(p))
        self.fit_names = sorted({item['name'] for p in train_payloads for item in p['history'] + p['candidates']})
        self.names.fit(self.fit_names)
        self.fit_group_keys = [p['user_alias'] for p in train_payloads]
        self.fit_cutoffs = sorted({p['cutoff'] for p in train_payloads})
        self.feature_count = len(self.dictionary.feature_names_) + 3 * len(self.names.vocabulary_) + 2
        return self

    def transform(self, payloads):
        blocks = []
        unknown_category_values = total_category_values = 0
        unknown_name_tokens = total_name_tokens = 0
        analyzer = self.names.build_analyzer()
        for p in payloads:
            rows = list(structured_rows(p))
            structured = self.dictionary.transform(rows)
            for row in rows:
                for k,v in row.items():
                    if isinstance(v,str):
                        total_category_values += 1
                        unknown_category_values += (k + '=' + v not in self.dictionary.vocabulary_)
            names = [i['name'] for i in p['candidates']]
            history_names = [i['name'] for i in p['history']]
            for name in names + history_names:
                terms = analyzer(name)
                total_name_tokens += len(terms)
                unknown_name_tokens += sum(t not in self.names.vocabulary_ for t in terms)
            candidate = self.names.transform(names)
            history = self.names.transform(history_names)
            if history_names:
                count_weights = np.asarray([h['buy_count'] for h in p['history']],dtype=float)
                recency_weights = np.asarray([1/(1+h['last_buy_age_days']) for h in p['history']],dtype=float)
                count_view = normalize(sparse.csr_matrix(history.multiply(count_weights[:,None]).sum(axis=0)))
                recency_view = normalize(sparse.csr_matrix(history.multiply(recency_weights[:,None]).sum(axis=0)))
            else:
                count_view = sparse.csr_matrix((1,history.shape[1]))
                recency_view = count_view.copy()
            count_repeated = sparse.vstack([count_view]*len(names),format='csr')
            recency_repeated = sparse.vstack([recency_view]*len(names),format='csr')
            overlaps = sparse.csr_matrix(np.column_stack([
                np.asarray(candidate.multiply(count_repeated).sum(axis=1)).ravel(),
                np.asarray(candidate.multiply(recency_repeated).sum(axis=1)).ravel()]))
            blocks.append(sparse.hstack([structured,candidate,count_repeated,recency_repeated,overlaps],format='csr'))
        result = sparse.vstack(blocks,format='csr')
        if result.shape[1] != self.feature_count or not np.isfinite(result.data).all():
            raise ValueError('Invalid feature matrix')
        self.last_transform_diagnostics = {'unknown_category_values':int(unknown_category_values),
            'total_category_values':int(total_category_values), 'unknown_name_tokens':int(unknown_name_tokens),
            'total_name_tokens':int(total_name_tokens)}
        return result

    def feature_names(self):
        terms = list(self.names.get_feature_names_out())
        return list(self.dictionary.get_feature_names_out()) + [prefix+t for prefix in ['candidate_name:','history_count_name:','history_recency_name:'] for t in terms] + ['candidate_history_count_cosine','candidate_history_recency_cosine']

    def fitted_state(self):
        return {'groups':self.fit_group_keys, 'names':self.fit_names,
                'dictionary':{k:int(v) for k,v in self.dictionary.vocabulary_.items()},
                'text':{k:int(v) for k,v in self.names.vocabulary_.items()},
                'idf':self.names.idf_.tolist()}
