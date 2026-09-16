import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from reviewpulse.predict import predict, load_production_model


class TestPredict(unittest.TestCase):
    def test_production_model_loads(self):
        model = load_production_model(models_dir="models")
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

    def test_predict_single_and_batch(self):
        pos_review = "This product is absolutely wonderful and high quality!"
        neg_review = "Terrible item, broke in one day and does not work."

        results = predict([pos_review, neg_review], models_dir="models")
        self.assertEqual(len(results), 2)

        for r in results:
            self.assertIn("text", r)
            self.assertIn("label", r)
            self.assertIn("sentiment", r)
            self.assertIn(r["label"], (0, 1))
            self.assertIn(r["sentiment"], ("Positive", "Negative"))
            if "confidence" in r:
                self.assertTrue(0.0 <= r["confidence"] <= 1.0)

        self.assertEqual(results[0]["sentiment"], "Positive")
        self.assertEqual(results[1]["sentiment"], "Negative")


if __name__ == "__main__":
    unittest.main()
