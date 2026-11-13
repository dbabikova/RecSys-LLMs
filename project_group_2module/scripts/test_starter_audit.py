"""Hand-checkable diagnostics tests using the imported official starter."""

import copy
import math
import unittest
from collections import Counter

from starter_audit import METHODS, ROOT, audit_cohort, import_starter, read_readme_references


class StarterAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = ROOT.parents[1] / "course-starter"
        cls.st = import_starter(cls.repo / "project" / "far-trans" / "starter.py")

    def fixture(self, all_repeat=False):
        st = self.st
        users = ["FIXTURE_U1", "FIXTURE_U2", "FIXTURE_U3"]
        assets = {
            chr(i): {
                "assetName": f"Fixture {chr(i)}", "assetCategory": "Stock",
                "assetSubCategory": "", "sector": "", "industry": "",
                "marketID": "X", "timestamp": "2020-01-01",
            } for i in range(ord("A"), ord("L") + 1)
        }
        profiles = {
            u: {"timestamp": "2020-01-01", "lastQuestionnaireDate": "2020-01-01"}
            for u in users
        }
        history = dict(zip(users, [Counter(A=2), Counter(L=1), Counter(D=1)]))
        truth = {users[0]: {"A"} if all_repeat else {"A", "L"}, users[1]: {"L"}}
        end_history = copy.deepcopy(history)
        end_history[users[0]].update({"A": 1} if all_repeat else {"A": 1, "L": 3})
        end_history[users[1]].update({"L": 2})
        features = {item: st.tokens(row) for item, row in assets.items()}
        pop = st.popularity_scores(history, assets)
        contexts, predictions = st.contexts_for(
            users, assets, profiles, history, features, pop, "2021-06-01", 10, False
        )
        events = {u: [("2021-05-28", item, "Buy") for item in counts] for u, counts in history.items()}
        recency = st.recency_rankings(contexts, pop, events)
        for u in users:
            predictions[u].update(recency[u])
        metrics = {
            method: st.aggregate([
                {"relevant_count": len(truth.get(u, set())),
                 "ndcg_at_10": st.ndcg10(predictions[u][method], truth.get(u, set()))}
                for u in users
            ]) for method in METHODS
        }
        return audit_cohort(
            st, users, contexts, predictions, history, truth, end_history,
            profiles, {"metrics": metrics},
        ), metrics

    def test_macro_micro_denominators_and_repeat_event_definition(self):
        d, _ = self.fixture()
        all10 = next(r for r in d["candidate_recall"] if r["target_group"] == "all" and r["k"] == 10)
        # U1 hits A of {A,L}; U2 misses L; U3 has empty truth and is not a recall denominator.
        self.assertEqual(all10["users"], 2)
        self.assertEqual(all10["target_pairs"], 3)
        self.assertAlmostEqual(all10["macro_recall"], .25)
        self.assertAlmostEqual(all10["micro_recall"], 1 / 3)
        self.assertEqual(all10["users_any_hit"], 1)
        self.assertAlmostEqual(d["repeat_pair_share"], 2 / 3)
        self.assertAlmostEqual(d["repeat_buy_row_share"], 3 / 6)

    def test_oracle_uses_full_truth_idcg_and_cannot_recover_missing_asset(self):
        d, metrics = self.fixture()
        expected_positive = (1 / (1 + 1 / math.log2(3))) / 2
        self.assertAlmostEqual(d["cbf50_oracle"]["ndcg_at_10_nonempty"], expected_positive)
        self.assertAlmostEqual(d["cbf50_oracle"]["ndcg_at_10"], expected_positive * 2 / 3)
        self.assertLess(d["cbf50_oracle"]["ndcg_at_10"], metrics["latest_buy"]["ndcg_at_10"])
        overlap = next(r for r in d["latest_buy_overlap"] if r["target_group"] == "all")
        self.assertEqual(overlap["latest_buy_hit_pairs"], 3)
        self.assertEqual(overlap["also_in_cbf50"], 1)
        self.assertEqual(overlap["missing_from_cbf50"], 2)

    def test_empty_target_subgroup_is_undefined_not_division_by_zero(self):
        d, _ = self.fixture(all_repeat=True)
        rows = [r for r in d["candidate_recall"] if r["target_group"] == "new"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["users"] == 0 and r["macro_recall"] is None for r in rows))

    def test_readme_parser_does_not_mix_demo_warm_and_all(self):
        references = read_readme_references(self.repo)
        self.assertEqual(references["demo"]["cbf"]["users"], 100)
        self.assertEqual(references["full_warm"]["cbf"]["users"], 10404)
        self.assertEqual(references["all"]["cbf"]["users"], 23571)


if __name__ == "__main__":
    unittest.main()
