import unittest
from ml_engine import evaluate_models, load_builtin

class TestModelComparison(unittest.TestCase):
    def test_classification(self):
        df, target, task = load_builtin('Iris (classification)')
        result, models, outputs, count, errors = evaluate_models(df, target, task)
        self.assertEqual(len(result), 6)
        self.assertEqual(count, 30)
        self.assertTrue(result['Accuracy'].between(0, 1).all())
        self.assertTrue(((result['Accuracy'] + result['Error Rate']) - 1).abs().max() < 1e-10)

    def test_regression(self):
        df, target, task = load_builtin('Diabetes (regression)')
        result, models, outputs, count, errors = evaluate_models(df, target, task)
        self.assertEqual(len(result), 6)
        self.assertEqual(count, 89)
        self.assertTrue((result['RMSE'] >= 0).all())

if __name__ == '__main__':
    unittest.main()
