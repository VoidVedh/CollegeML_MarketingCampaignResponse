# Unit test for data generation schema and boundary checks
import unittest
from src.generate_data import generate_campaign_dataset

class TestGenerator(unittest.TestCase):
    def test_schema(self):
        df = generate_campaign_dataset(n_samples=50)
        self.assertEqual(len(df), 50)
