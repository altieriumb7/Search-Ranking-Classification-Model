import unittest

from src.data_loader import Document, Qrel, Query, validate_dataset


class DataValidationTest(unittest.TestCase):
    def test_validate_dataset_rejects_unknown_qrel_reference(self):
        queries = [Query("q1", "reset password", "train")]
        documents = [Document("d1", "Reset", "How to reset", "help")]
        qrels = [Qrel("q1", "d2", 2)]

        with self.assertRaises(ValueError):
            validate_dataset(queries, documents, qrels)

    def test_validate_dataset_accepts_valid_minimal_dataset(self):
        queries = [Query("q1", "reset password", "test")]
        documents = [Document("d1", "Reset", "How to reset", "help")]
        qrels = [Qrel("q1", "d1", 1)]

        validate_dataset(queries, documents, qrels)


if __name__ == "__main__":
    unittest.main()
