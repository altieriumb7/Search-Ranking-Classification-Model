import unittest

from src.data_loader import load_dataset
from src.features import FeatureExtractor
from src.model import LearningToRankModel
from src.ranking import RankingPipeline, build_training_matrix


class PredictionOutputTest(unittest.TestCase):
    def test_prediction_rows_have_demo_contract(self):
        queries, documents, qrels = load_dataset()
        train_queries = [query for query in queries if query.split == "train"][:2]
        extractor = FeatureExtractor().fit(documents)
        X, y, qids, _doc_ids, groups = build_training_matrix(
            train_queries, documents, qrels, extractor
        )
        ranker = LearningToRankModel(prefer_xgboost=False).fit(X, y, qids, groups)
        pipeline = RankingPipeline(documents, extractor, ranker)

        rows = pipeline.rank_learned("password reset email not received", top_k=3)

        self.assertEqual(len(rows), 3)
        for index, row in enumerate(rows, start=1):
            self.assertEqual(row["rank"], index)
            self.assertIsInstance(row["doc_id"], str)
            self.assertIsInstance(row["title"], str)
            self.assertIsInstance(row["score"], float)


if __name__ == "__main__":
    unittest.main()
