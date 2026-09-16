import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from reviewpulse.preprocessing import TextCleaner, load_and_prepare_data


class TestPreprocessing(unittest.TestCase):
    def test_text_cleaner_basic(self):
        cleaner = TextCleaner()
        raw = ["Hello World! 123", "THIS IS A TEST...", "Awesome & Great!"]
        cleaned = cleaner.transform(raw)
        self.assertEqual(list(cleaned), ["hello world 123", "this is a test", "awesome  great"])

    def test_text_cleaner_series_and_df(self):
        cleaner = TextCleaner()
        series = pd.Series(["Product #1: Superb!!"])
        df = pd.DataFrame({"text": ["Product #2: Horrible!!"]})

        self.assertEqual(cleaner.transform(series).iloc[0], "product 1 superb")
        self.assertEqual(cleaner.transform(df).iloc[0], "product 2 horrible")

    def test_load_and_prepare_data_fasttext_labels(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("label,text\n1,bad product\n2,good product\n1,terrible\n2,excellent\n")
            filepath = f.name

        try:
            X, y = load_and_prepare_data(filepath, sample_size=10)
            self.assertEqual(len(X), 4)
            # 1 -> Negative (0), 2 -> Positive (1)
            self.assertEqual(list(y), [0, 1, 0, 1])
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
