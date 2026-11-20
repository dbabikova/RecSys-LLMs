import math
import unittest
from ranker_metrics import evaluate


class MetricTests(unittest.TestCase):
    def fixture(self,truth_count=2):
        g={'group_key':'G','labels':[1]+[0]*19,'full_truth_count':truth_count,'candidate_aliases':[f'A{i:02d}' for i in range(20)]}
        m={'group_key':'G','pool_isins':[f'I{i:02d}' for i in range(20)]}
        return g,m

    def test_full_truth_denominator(self):
        g,m=self.fixture()
        metric,_,_=evaluate(list(range(20,0,-1)),[g],[m])
        self.assertAlmostEqual(metric['ndcg_at_10'],1/(1+1/math.log2(3)))
        self.assertLess(metric['ndcg_at_10'],1)

    def test_empty_truth_is_zero_and_retained(self):
        g,m=self.fixture(0)
        g['labels']=[0]*20
        metric,_,_=evaluate([0]*20,[g],[m])
        self.assertEqual(metric['users'],1)
        self.assertEqual(metric['ndcg_at_10'],0)
        self.assertEqual(metric['nonempty_truth_users'],0)

    def test_unretrieved_truth_does_not_disappear(self):
        g,m=self.fixture(2)
        g['labels']=[0]*20
        metric,rows,_=evaluate([0]*20,[g],[m])
        self.assertEqual(metric['nonempty_truth_users'],1)
        self.assertEqual(rows[0]['full_truth_count'],2)
        self.assertEqual(metric['ndcg_at_10'],0)

    def test_ties_use_isin_not_input_order(self):
        g,m=self.fixture()
        m['pool_isins']=list(reversed(m['pool_isins']))
        _,_,p=evaluate([0]*20,[g],[m])
        self.assertEqual([r['ISIN'] for r in p],sorted(m['pool_isins'])[:10])

    def test_nonfinite_or_wrong_length_rejected(self):
        g,m=self.fixture()
        for scores in [[float('nan')]*20,[0]*19]:
            with self.assertRaisesRegex(ValueError,'Invalid prediction'):
                evaluate(scores,[g],[m])


if __name__=='__main__':
    unittest.main()
