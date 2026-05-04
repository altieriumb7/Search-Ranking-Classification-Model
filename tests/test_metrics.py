import unittest

from src.metrics import average_precision, evaluate_rankings, ndcg_at_k, reciprocal_rank


class MetricsTest(unittest.TestCase):
    def test_ranking_metrics_known_values(self):
        relevances = [3, 2, 0, 1]

        self.assertAlmostEqual(ndcg_at_k(relevances, k=4), 0.9926, places=3)
        self.assertAlmostEqual(average_precision(relevances), 0.9167, places=3)
        self.assertEqual(reciprocal_rank(relevances), 1.0)

    def test_evaluate_rankings(self):
        qrels = {"q1": {"d1": 3, "d2": 0, "d3": 1}}
        rankings = {"q1": ["d3", "d2", "d1"]}
        metrics = evaluate_rankings(qrels, rankings, k=3)

        self.assertIn("ndcg@3", metrics)
        self.assertIn("map", metrics)
        self.assertIn("mrr", metrics)


if __name__ == "__main__":
    unittest.main()
