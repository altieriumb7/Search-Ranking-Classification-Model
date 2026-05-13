import unittest
from pathlib import Path

from src.model import load_pickle


class ModelSecurityTest(unittest.TestCase):
    def test_load_pickle_rejects_paths_outside_models_directory(self):
        with self.assertRaises(ValueError):
            load_pickle(Path("artifact.pkl"))


if __name__ == "__main__":
    unittest.main()
