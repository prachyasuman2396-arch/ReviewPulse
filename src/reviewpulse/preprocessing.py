import os
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to clean text data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a pandas Series or 1D array
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif not isinstance(X, pd.Series):
            X = pd.Series(X)

        # Lowercase, remove non-alphanumeric characters, and strip extra whitespace
        return (
            X.astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
            .str.strip()
        )


def load_and_prepare_data(filepath, sample_size=10000, random_state=42):
    """
    Loads review data and creates binary target (1=Positive, 0=Negative).
    
    Supports:
      - CSV files (with 'label' and 'text' columns)
      - FastText .ft.txt files
      - Amazon Polarity labels: 1 -> 0 (Negative), 2 -> 1 (Positive)
      - Star rating labels (1-5): 1,2 -> 0, 4,5 -> 1 (drops 3)
    """
    if filepath.endswith(".ft.txt") or (filepath.endswith(".txt") and not os.path.exists(filepath.replace(".txt", ".csv"))):
        labels = []
        texts = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    labels.append(int(parts[0].replace("__label__", "").strip()))
                    texts.append(parts[1].strip())
        df = pd.DataFrame({"label": labels, "text": texts})
    else:
        df = pd.read_csv(filepath)

    # Drop nulls
    df = df.dropna(subset=["text", "label"])

    # Map labels to binary sentiment (0=Negative, 1=Positive)
    unique_labels = set(df["label"].unique())
    if unique_labels.issubset({1, 2}):
        # Amazon Polarity format: 1 is Negative, 2 is Positive
        df["sentiment"] = df["label"].map({1: 0, 2: 1})
    else:
        # Standard 1-5 star ratings: drop neutral (3)
        df = df[df["label"] != 3]
        df["sentiment"] = (df["label"] > 3).astype(int)

    # Downsample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    return df["text"].reset_index(drop=True), df["sentiment"].reset_index(drop=True)