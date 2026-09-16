import json
import os
import sys
import joblib

# Ensure the package root is in sys.path for pickle deserialization
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import mlflow.pyfunc
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def load_production_model(models_dir: str = "models", from_registry: bool = False, tracking_uri: str | None = None):
    """
    Loads the production champion model.
    By default, loads from the local models/ directory for zero-latency offline inference.
    If from_registry=True, loads directly from MLflow Model Registry via alias @champion.
    """
    if from_registry:
        if not HAS_MLFLOW:
            raise ImportError("mlflow is required to load from registry.")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            import mlflow.sklearn
            return mlflow.sklearn.load_model("models:/ReviewPulseModel@champion")
        except Exception:
            return mlflow.pyfunc.load_model("models:/ReviewPulseModel@champion")

    local_path = os.path.join(models_dir, "model.joblib")
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Production model artifact not found at '{local_path}'. "
            f"Run 'python src/reviewpulse/train.py' first to generate it."
        )
    return joblib.load(local_path)


def predict(texts: str | list[str], models_dir: str = "models", from_registry: bool = False) -> list[dict]:
    """
    Predicts sentiment for one or multiple texts.
    Returns structured results with label, sentiment name, and confidence score.
    """
    model = load_production_model(models_dir=models_dir, from_registry=from_registry)

    if isinstance(texts, str):
        single_input = True
        texts = [texts]
    else:
        single_input = False

    predictions = model.predict(texts)

    # Determine probabilities if model supports them
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)[:, 1]

    results = []
    for idx, text in enumerate(texts):
        label = int(predictions[idx])
        sentiment = "Positive" if label == 1 else "Negative"
        entry = {
            "text": text,
            "label": label,
            "sentiment": sentiment,
        }
        if probs is not None:
            confidence = float(probs[idx]) if label == 1 else float(1.0 - probs[idx])
            entry["confidence"] = round(confidence, 4)
            entry["positive_probability"] = round(float(probs[idx]), 4)
        results.append(entry)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReviewPulse Sentiment Inference CLI")
    parser.add_argument(
        "text",
        nargs="*",
        type=str,
        help="Review text(s) to classify. If omitted, runs default samples.",
    )
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="Load champion model directly from MLflow Model Registry.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing the local model.joblib artifact.",
    )

    args = parser.parse_args()

    if args.text:
        input_texts = args.text
    else:
        input_texts = [
            "This product is absolutely amazing! Exceeded all my expectations.",
            "Terrible purchase. It broke within 2 days and customer service was unhelpful.",
            "Decent quality for the price, works as expected.",
        ]

    mode = "MLflow Model Registry (@champion)" if args.from_registry else f"Local Artifact ({args.models_dir}/model.joblib)"
    print(f"\n[ReviewPulse Inference] Source: {mode}")
    print("-" * 75)

    predictions = predict(input_texts, models_dir=args.models_dir, from_registry=args.from_registry)
    for p in predictions:
        conf_str = f"{p['confidence']*100:.1f}%" if "confidence" in p else "N/A"
        badge = "🟢 POSITIVE" if p["label"] == 1 else "🔴 NEGATIVE"
        print(f"{badge} (Confidence: {conf_str})")
        print(f"  Review: \"{p['text']}\"\n")

