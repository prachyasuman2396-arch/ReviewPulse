import os
import mlflow

# Use sqlite:///mlflow.db to avoid macOS port 5000 AirPlay collisions
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("ReviewPulse_Experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("C", 1.0)

    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.84)

    print("MLflow run completed!")