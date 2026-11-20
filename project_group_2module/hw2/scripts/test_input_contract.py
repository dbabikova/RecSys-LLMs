"""Safety boundaries and presentation invariance on controlled past-only fixtures."""
import unittest
from input_contract import common_payload, feature_records, presentation, profile_input


class ContractTests(unittest.TestCase):
    def fixture(self):
        assets = {f'I{i:02d}': {'assetName':f'Asset {i}', 'timestamp':'2020-01-01',
                  'assetCategory':'Stock', 'assetSubCategory':'Large', 'sector':'X',
                  'industry':'Y','marketID':'M'} for i in range(21)}
        counts = {i:1 for i in assets}
        dates = {i:'2021-05-01' for i in assets}
        profile = {'timestamp':'2021-01-01','lastQuestionnaireDate':'2020-01-01',
                   'customerType':'Mass','riskLevel':'Predicted_Balanced','investmentCapacity':'CAP_LT_30K'}
        return assets, counts, dates, profile

    def test_future_or_missing_questionnaire_hidden(self):
        *_, p = self.fixture()
        for value in ['2021-06-01','2022-01-01','2000-01-01','','bad-date']:
            q = profile_input(dict(p,lastQuestionnaireDate=value),'2021-06-01')
            self.assertEqual(q['riskLevel']['value'],'[MISSING]')
            self.assertEqual(q['investmentCapacity']['status'],'withheld_unverified')
            self.assertEqual(q['customerType']['value'],'Mass')
        self.assertEqual(profile_input(p,'2021-06-01')['riskLevel']['status'],'estimated')

    def test_future_snapshot_hidden(self):
        *_, p = self.fixture()
        q = profile_input(dict(p,timestamp='2021-06-01'),'2021-06-01')
        self.assertTrue(all(q[f]['value']=='[MISSING]' for f in ['customerType','riskLevel','investmentCapacity']))

    def test_future_metadata_or_buy_rejected(self):
        a,c,d,p = self.fixture()
        a['I00']['timestamp']='2021-06-01'
        with self.assertRaisesRegex(ValueError,'Future metadata'):
            common_payload('U001','2021-06-01',a,p,c,d,list(a)[:20])
        a['I00']['timestamp']='2020-01-01'
        d['I00']='2021-06-01'
        with self.assertRaisesRegex(ValueError,'future Buy'):
            common_payload('U001','2021-06-01',a,p,c,d,list(a)[:20])

    def test_truncation_and_permutation_change_only_order(self):
        a,c,d,p = self.fixture()
        base = common_payload('U001','2021-06-01',a,p,c,d,list(a)[:20])
        self.assertEqual(base['history_summary']['omitted_distinct_assets'],1)
        rows = {r['candidate_alias']:r for r in feature_records(base)}
        for seed in [17,42,2026]:
            other = presentation(base,seed)
            self.assertEqual({k:v for k,v in other.items() if k!='candidates'}, {k:v for k,v in base.items() if k!='candidates'})
            self.assertEqual({r['asset_alias']:r for r in other['candidates']},{r['asset_alias']:r for r in base['candidates']})
            self.assertEqual({r['candidate_alias']:r for r in feature_records(other)},rows)
            self.assertEqual(other,presentation(base,seed))


if __name__ == '__main__':
    unittest.main()
