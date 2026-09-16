import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from reviewpulse.insights import generate_amazon_review_insights, print_amazon_insights_card


class TestAmazonInsights(unittest.TestCase):
    def test_amazon_insights_structure(self):
        sample_reviews = [
            "Fantastic sound quality and long battery life.",
            "Very comfortable to wear for hours.",
            "Died after 3 months of use, customer service was awful.",
            "Great value for money.",
        ]

        insights = generate_amazon_review_insights(sample_reviews, product_id="TEST-HEADPHONES-1")
        self.assertIn("summary", insights)
        self.assertIn("overall_sentiment", insights)
        self.assertIn(insights["overall_sentiment"], ("Positive", "Mixed", "Negative"))
        self.assertIn("average_rating_estimate", insights)
        self.assertTrue(1.0 <= insights["average_rating_estimate"] <= 5.0)
        self.assertIn("what_customers_liked", insights)
        self.assertIsInstance(insights["what_customers_liked"], list)
        self.assertTrue(len(insights["what_customers_liked"]) > 0)
        self.assertIn("what_customers_disliked", insights)
        self.assertIsInstance(insights["what_customers_disliked"], list)
        self.assertIn("aspect_breakdown", insights)
        self.assertIsInstance(insights["aspect_breakdown"], dict)

    def test_amazon_insights_empty_raises(self):
        with self.assertRaises(ValueError):
            generate_amazon_review_insights([])

    def test_amazon_insights_card_printer(self):
        sample_reviews = ["Great product, loved it!"]
        insights = generate_amazon_review_insights(sample_reviews)
        # Ensure printer runs cleanly without error
        print_amazon_insights_card(insights)


if __name__ == "__main__":
    unittest.main()
