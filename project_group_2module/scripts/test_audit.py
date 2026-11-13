"""Small, hand-checkable tests; no external data or customer information."""

import math
import unittest

import pandas as pd

from audit_core import evaluate_split, latest, repeat_stats


class AuditTests(unittest.TestCase):
    def test_repeat_definition_does_not_depend_on_file_order(self):
        f = pd.DataFrame({
            "customerID": ["a", "a", "a", "b"],
            "ISIN": ["i", "i", "i", "i"],
            "timestamp": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-01", "2020-01-02"]),
        })
        result = repeat_stats(f)
        self.assertEqual(result["beyond_first_row"], 2)
        self.assertEqual(result["strictly_later_day"], 1)
        self.assertEqual(result["extra_same_first_day"], 1)

    def test_asof_does_not_choose_future_snapshot(self):
        f = pd.DataFrame({
            "customerID": ["a", "a", "b"],
            "timestamp": pd.to_datetime(["2020-01-01", "2022-01-01", "2022-01-01"]),
            "value": ["old", "future", "future"],
        })
        result = latest(f, "customerID", pd.Timestamp("2021-01-01"))
        self.assertEqual(result.value.tolist(), ["old"])

    def fixture(self):
        cut, end = pd.Timestamp("2022-05-29"), pd.Timestamp("2022-11-29")
        tx = pd.DataFrame([
            ("a", "I01", 1, "Buy", cut),       # Include cut in train.
            ("b", "I02", 2, "Buy", cut),
            ("a", "I01", 3, "Buy", end),       # Include end in labels B.
            ("a", "I03", 4, "Buy", end),
            ("a", "I03", 5, "Buy", end),       # Binary relevance, not count.
            ("b", "I04", 6, "Buy", end),
            ("cold", "I05", 7, "Buy", end),
            ("a", "I06", -1, "Buy", end),      # Never a label.
            ("a", "I07", 8, "Sell", end),      # Never a label.
            ("b", "I08", 9, "Buy", end + pd.Timedelta(days=1)),
        ], columns=["customerID", "ISIN", "transactionID", "transactionType", "timestamp"])
        return {"transactions": tx}, {
            "cut": cut, "end": end, "eligible": {f"I{i:02}" for i in range(1, 21)}
        }

    def test_boundary_filters_binary_labels_and_hand_ndcg(self):
        tables, stats = self.fixture()
        rows, details = evaluate_split(tables, stats)
        a, b = rows
        self.assertEqual(a["users"], 2)
        self.assertEqual(a["cold_users"], 1)
        self.assertEqual(a["relevant_pairs"], 2)
        self.assertEqual(b["relevant_pairs"], 3)
        self.assertEqual(details["A"]["relevance"]["a"], {"I03"})
        self.assertEqual(details["B"]["relevance"]["a"], {"I01", "I03"})
        self.assertEqual(details["B"]["relevance"]["b"], {"I04"})
        # A: a has I03 at rank 2 after masking I01; b has I04 at rank 3.
        expected_a = (1 / math.log2(3) + 1 / math.log2(4)) / 2
        # B: a has I01 rank 1 and I03 rank 3; b has I04 rank 4.
        expected_b = ((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
                      + 1 / math.log2(5)) / 2
        self.assertAlmostEqual(a["ndcg_popularity"], expected_a)
        self.assertEqual(a["ndcg_popularity"], a["ndcg_repeat_popularity"])
        self.assertAlmostEqual(b["ndcg_popularity"], expected_b)

    def test_reconstruction_removal_changes_warm_population(self):
        tables, stats = self.fixture()
        tables["transactions"] = pd.concat([
            tables["transactions"],
            pd.DataFrame([
                ("reconstructed_only", "I09", -2, "Buy", stats["cut"]),
                ("reconstructed_only", "I10", 11, "Buy", stats["end"]),
            ], columns=tables["transactions"].columns),
        ], ignore_index=True)
        with_negative, _ = evaluate_split(tables, stats)
        without_negative, _ = evaluate_split(tables, stats, include_reconstructed=False)
        self.assertEqual(with_negative[0]["users"], without_negative[0]["users"] + 1)
        self.assertEqual(with_negative[0]["cold_users"] + 1, without_negative[0]["cold_users"])


def run_checks():
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    )
    if not result.wasSuccessful():
        raise AssertionError("Audit self-tests failed")


if __name__ == "__main__":
    run_checks()
