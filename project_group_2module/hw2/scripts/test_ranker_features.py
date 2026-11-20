import copy
import unittest
import numpy as np

from input_contract import canonical, digest, presentation
from ranker_features import RankerFeatures


class FeatureTests(unittest.TestCase):
    def payload(self):
        item={'asset_alias':'A001','name':'alpha bank','attributes':{f:'known' for f in ['assetCategory','assetSubCategory','sector','industry','marketID']},'metadata_age_days':100}
        return {'user_alias':'train:U001','cutoff':'2021-01-01','repeat_purchases_allowed':True,
                'profile':{'snapshot_age_days':2, **{f:{'value':'known','status':'observed'} for f in ['customerType','riskLevel','investmentCapacity']}},
                'history_summary':{'full_distinct_assets':1,'full_buy_events':2,'retained_distinct_assets':1,'retained_buy_events':2,'omitted_distinct_assets':0},
                'history':[dict(item,buy_count=2,last_buy_age_days=1)],
                'candidates':[dict(item,asset_alias=f'A{i:03d}',source_cbf_rank=i,source_cbf_score=0.5,past_buy_count=1,last_buy_age_days=2) for i in range(1,21)]}

    def encoder(self,p):
        return RankerFeatures({'analyzer':'word','ngram_range':[1,2],'max_features':4096,'lowercase':True,'smooth_idf':True}).fit([p])

    def test_validation_unknown_values_do_not_fit(self):
        p=self.payload()
        encoder=self.encoder(p)
        before=digest(canonical(encoder.fitted_state()))
        other=copy.deepcopy(p)
        other['candidates'][0]['name']='validationonlytoken'
        other['candidates'][0]['attributes']['sector']='validationonlysector'
        x=encoder.transform([other])
        self.assertEqual(x.shape[0],20)
        self.assertNotIn('validationonlytoken',encoder.names.vocabulary_)
        self.assertNotIn('c:sector=validationonlysector',encoder.dictionary.vocabulary_)
        self.assertGreater(encoder.last_transform_diagnostics['unknown_category_values'],0)
        self.assertEqual(before,digest(canonical(encoder.fitted_state())))

    def test_permutation_only_reorders_matrix_rows(self):
        p=self.payload()
        encoder=self.encoder(p)
        x=encoder.transform([p])
        other=presentation(p,17)
        y=encoder.transform([other])
        positions={c['asset_alias']:i for i,c in enumerate(p['candidates'])}
        expected=x[[positions[c['asset_alias']] for c in other['candidates']]]
        self.assertEqual((expected-y).nnz,0)

    def test_labels_and_identity_are_not_features(self):
        p=self.payload()
        encoder=self.encoder(p)
        x=encoder.transform([p])
        other=copy.deepcopy(p)
        other['user_alias']='validation:OTHER'
        other['labels']=[1]*20
        y=encoder.transform([other])
        self.assertEqual((x-y).nnz,0)
        self.assertTrue(np.isfinite(y.data).all())


if __name__=='__main__':
    unittest.main()
