import unittest

from src.data_loader import load_dataset


class DataLoadingTest(unittest.TestCase):
    def test_demo_dataset_loads(self):
        queries, documents, qrels = load_dataset()

        self.assertGreaterEqual(len(queries), 10)
        self.assertGreaterEqual(len(documents), 30)
        self.assertGreaterEqual(len(qrels), 80)
        self.assertTrue({query.split for query in queries} >= {"train", "test"})


if __name__ == "__main__":
    unittest.main()
