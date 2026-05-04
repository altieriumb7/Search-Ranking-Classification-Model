import unittest

from src.data_loader import load_documents
from src.features import FeatureExtractor


class FeatureExtractionTest(unittest.TestCase):
    def test_matching_document_has_positive_bm25_and_feature_shape(self):
        documents = load_documents()
        extractor = FeatureExtractor().fit(documents)
        target = next(document for document in documents if document.doc_id == "d025")

        features = extractor.transform_pair("french press coffee grind", target)

        self.assertEqual(len(features), len(extractor.feature_names))
        self.assertGreater(features[0], 0.0)
        self.assertGreaterEqual(features[2], 0.5)


if __name__ == "__main__":
    unittest.main()
