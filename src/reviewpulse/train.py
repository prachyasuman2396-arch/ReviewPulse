import argparse
import datetime
import json
import os
import sys
import urllib.request
import warnings
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Support running directly or as an installed package
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PACKAGE_DIR, "..")))
try:
    from reviewpulse.preprocessing import TextCleaner, load_and_prepare_data
except ModuleNotFoundError:
    # pyrefly: ignore [missing-import]
    from preprocessing import TextCleaner, load_and_prepare_data


def get_default_tracking_uri() -> str:
    """
    Returns an appropriate MLflow tracking URI.
    Checks if a local MLflow server is active on port 5000 or 5001,
    or falls back to sqlite:///mlflow.db.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    for url in ["http://127.0.0.1:5000", "http://127.0.0.1:5001"]:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            pass

    return "sqlite:///mlflow.db"


def resolve_data_path(provided_path: str | None = None) -> str:
    """Finds the dataset file, checking common locations."""
    if provided_path and os.path.exists(provided_path):
        return provided_path

    candidates = [
        "data/raw/test.csv",
        "data/raw/test.ft.txt",
        "data/Reviews.csv",
        "data/processed/test.csv",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No dataset found. Tried: {provided_path} and standard paths {candidates}"
    )


def build_experiment_configs():
    """
    Defines diverse production-grade models and hyperparameter grids:
      - Logistic Regression (varying C and TF-IDF features)
      - Linear Support Vector Machine (LinearSVC with CalibratedClassifierCV for probabilities)
      - Multinomial Naive Bayes (varying smoothing alpha)
    """
    return [
        # --- Logistic Regression Models ---
        {
            "run_name": "logreg_c0.5_tfidf5k",
            "model_type": "LogisticRegression",
            "clf": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
            "model_params": {"C": 0.5, "penalty": "l2", "solver": "lbfgs"},
            "tfidf_params": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
        {
            "run_name": "logreg_c1.0_tfidf10k",
            "model_type": "LogisticRegression",
            "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "model_params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
            "tfidf_params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
        {
            "run_name": "logreg_c2.0_tfidf10k",
            "model_type": "LogisticRegression",
            "clf": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
            "model_params": {"C": 2.0, "penalty": "l2", "solver": "lbfgs"},
            "tfidf_params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
        # --- Support Vector Machine (LinearSVC + CalibratedClassifierCV) ---
        {
            "run_name": "svm_c0.5_calibrated_tfidf5k",
            "model_type": "LinearSVM",
            "clf": CalibratedClassifierCV(
                LinearSVC(C=0.5, random_state=42, max_iter=2000), cv=3
            ),
            "model_params": {
                "C": 0.5,
                "base_estimator": "LinearSVC",
                "calibrated": True,
            },
            "tfidf_params": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
        {
            "run_name": "svm_c1.0_calibrated_tfidf10k",
            "model_type": "LinearSVM",
            "clf": CalibratedClassifierCV(
                LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=3
            ),
            "model_params": {
                "C": 1.0,
                "base_estimator": "LinearSVC",
                "calibrated": True,
            },
            "tfidf_params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
        # --- Multinomial Naive Bayes Models ---
        {
            "run_name": "nb_alpha0.5_tfidf5k",
            "model_type": "MultinomialNB",
            "clf": MultinomialNB(alpha=0.5),
            "model_params": {"alpha": 0.5},
            "tfidf_params": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
        {
            "run_name": "nb_alpha1.0_tfidf10k",
            "model_type": "MultinomialNB",
            "clf": MultinomialNB(alpha=1.0),
            "model_params": {"alpha": 1.0},
            "tfidf_params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "stop_words": "english",
            },
        },
    ]


def train_and_evaluate(
    data_path: str | None = None,
    sample_size: int = 15000,
    experiment_name: str = "ReviewPulse_Experiment",
    registered_model_name: str = "ReviewPulseModel",
    tracking_uri: str | None = None,
    models_dir: str = "models",
):
    """
    Loads data, trains multiple candidate models, logs results to MLflow,
    promotes the best model in the MLflow Model Registry, and exports
    the production model artifact and metadata to models/ directory.
    """
    resolved_path = resolve_data_path(data_path)
    uri = tracking_uri or get_default_tracking_uri()

    print(f"[ReviewPulse] Using Tracking URI: {uri}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    print(f"[ReviewPulse] Loading dataset from: {resolved_path}")
    actual_sample_size = sample_size if sample_size > 0 else None
    X, y = load_and_prepare_data(resolved_path, sample_size=actual_sample_size)
    print(f"[ReviewPulse] Loaded {len(X):,} reviews. Class distribution:")
    print(y.value_counts(normalize=True).to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[ReviewPulse] Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    configs = build_experiment_configs()
    results = []
    trained_pipelines = {}

    # Bundle preprocessing source code with MLflow model so it loads anywhere without dependency errors
    code_paths = [os.path.join(PACKAGE_DIR, "preprocessing.py")]
    if os.path.exists(os.path.join(PACKAGE_DIR, "__init__.py")):
        code_paths.append(PACKAGE_DIR)

    print(f"\n[ReviewPulse] Training and evaluating {len(configs)} models with MLflow tracking...")

    for idx, cfg in enumerate(configs, 1):
        run_name = cfg["run_name"]
        model_type = cfg["model_type"]
        print(f"\n[{idx}/{len(configs)}] Running experiment: {run_name} ({model_type})...")

        # Construct production pipeline
        pipeline = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(**cfg["tfidf_params"])),
            ("clf", cfg["clf"]),
        ])

        with mlflow.start_run(run_name=run_name) as run:
            # 1. Fit pipeline
            pipeline.fit(X_train, y_train)

            # 2. Evaluate
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                y_prob = pipeline.decision_function(X_test)
            else:
                y_prob = None

            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            roc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0

            # 3. Log Parameters to MLflow
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            for k, v in cfg["model_params"].items():
                mlflow.log_param(f"model__{k}", v)
            for k, v in cfg["tfidf_params"].items():
                mlflow.log_param(f"tfidf__{k}", str(v))

            # 4. Log Metrics to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", roc)

            # 5. Log Pipeline Model Artifact with bundled source code
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                code_paths=code_paths,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            results.append({
                "run_id": run.info.run_id,
                "run_name": run_name,
                "model_type": model_type,
                "model_uri": model_info.model_uri,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "roc_auc": round(roc, 4),
            })
            trained_pipelines[run.info.run_id] = pipeline

            print(f"    -> Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: {roc:.4f}")

    # Summarize and rank runs
    results_df = pd.DataFrame(results).sort_values(by=["f1_score", "roc_auc", "accuracy"], ascending=False).reset_index(drop=True)
    print("\n" + "=" * 75)
    print("                           MLFLOW LEADERBOARD")
    print("=" * 75)
    print(results_df[["run_name", "model_type", "f1_score", "accuracy", "roc_auc"]].to_string(index=False))

    best_run = results_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_model_uri = best_run["model_uri"]
    best_pipeline = trained_pipelines[best_run_id]

    print("\n" + "=" * 75)
    print(f" 🏆 BEST MODEL CHOSEN: {best_run['run_name']} ({best_run['model_type']})")
    print(f" Best F1: {best_run['f1_score']:.4f} | Accuracy: {best_run['accuracy']:.4f} | ROC-AUC: {best_run['roc_auc']:.4f}")
    print("=" * 75)

    # -------------------------------------------------------------
    # Production Step 1: Register in MLflow Model Registry
    # -------------------------------------------------------------
    print(f"\n[ReviewPulse] Registering best model from {best_model_uri} to registry '{registered_model_name}'...")
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    reg_version = mlflow.register_model(
        model_uri=best_model_uri,
        name=registered_model_name,
    )
    version_str = str(reg_version.version)

    # -------------------------------------------------------------
    # Production Step 2: Transition to Production & Set Champion Aliases
    # -------------------------------------------------------------
    print(f"[ReviewPulse] Promoting version {version_str} to 'Production' with alias '@champion'...")
    try:
        client.set_registered_model_alias(registered_model_name, "champion", version_str)
        client.set_registered_model_alias(registered_model_name, "production", version_str)
    except Exception as e:
        print(f"[ReviewPulse] Notice setting alias: {e}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=version_str,
                stage="Production",
                archive_existing_versions=True,
            )
    except Exception as e:
        print(f"[ReviewPulse] Notice setting stage: {e}")

    # Set rich production tags and description
    desc = (
        f"Production Champion Model ({best_run['model_type']}).\n"
        f"Selected from automated multi-model leaderboard.\n"
        f"F1-Score: {best_run['f1_score']:.4f} | Accuracy: {best_run['accuracy']:.4f} | ROC-AUC: {best_run['roc_auc']:.4f}"
    )
    try:
        client.update_model_version(registered_model_name, version_str, description=desc)
        client.set_model_version_tag(registered_model_name, version_str, "status", "champion")
        client.set_model_version_tag(registered_model_name, version_str, "f1_score", str(best_run["f1_score"]))
        client.set_model_version_tag(registered_model_name, version_str, "accuracy", str(best_run["accuracy"]))
        client.set_model_version_tag(registered_model_name, version_str, "roc_auc", str(best_run["roc_auc"]))
        client.set_model_version_tag(registered_model_name, version_str, "model_type", best_run["model_type"])
    except Exception as e:
        print(f"[ReviewPulse] Notice setting tags: {e}")

    # -------------------------------------------------------------
    # Production Step 3: Export Local Standalone Production Artifact
    # -------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    prod_model_path = os.path.join(models_dir, "model.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_pipeline, prod_model_path)

    metadata = {
        "model_name": registered_model_name,
        "model_version": version_str,
        "run_id": best_run_id,
        "run_name": best_run["run_name"],
        "model_type": best_run["model_type"],
        "metrics": {
            "f1_score": float(best_run["f1_score"]),
            "accuracy": float(best_run["accuracy"]),
            "precision": float(best_run["precision"]),
            "recall": float(best_run["recall"]),
            "roc_auc": float(best_run["roc_auc"]),
        },
        "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tracking_uri": uri,
        "stage": "Production",
        "alias": "champion",
        "artifacts": {
            "local_model_path": prod_model_path,
            "mlflow_model_uri": best_model_uri,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ReviewPulse] ✅ Model successfully registered in MLflow Model Registry as '{registered_model_name}' (Version {version_str})!")
    print(f"[ReviewPulse] ✅ Production artifact saved to: {prod_model_path}")
    print(f"[ReviewPulse] ✅ Production metadata saved to: {metadata_path}")
    print("=" * 75)

    return results_df, best_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ReviewPulse sentiment models with MLflow.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw/processed CSV or TXT dataset.")
    parser.add_argument("--sample-size", type=int, default=15000, help="Number of reviews to sample (0 for full dataset).")
    parser.add_argument("--experiment-name", type=str, default="ReviewPulse_Experiment", help="MLflow experiment name.")
    parser.add_argument("--registered-model-name", type=str, default="ReviewPulseModel", help="Model registry name.")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save production artifacts.")

    args = parser.parse_args()
    train_and_evaluate(
        data_path=args.data_path,
        sample_size=args.sample_size,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        tracking_uri=args.tracking_uri,
        models_dir=args.models_dir,
    )
