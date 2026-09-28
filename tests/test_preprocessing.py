# Verification test for preprocessing pipeline without data leakage
import unittest
from src.preprocess import create_preprocessor, load_and_split_data

class TestPreprocess(unittest.TestCase):
    def test_pipeline(self):
        X_train, X_test, y_train, y_test = load_and_split_data()
        prep = create_preprocessor()
        X_trans = prep.fit_transform(X_train)
        self.assertEqual(X_trans.shape[1], 11)
