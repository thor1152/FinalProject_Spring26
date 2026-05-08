import os
import json
import joblib
import pandas as pd
import boto3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(
    df: pd.DataFrame, 
    model_path: str = "models/breast_cancer_model.pkl", 
    metrics_path: str = "eval/breast_cancer_metrics.json",
    metadata_path: str = "models/breast_cancer_metadata.json"
    ) -> float:
    """Train a logistic regression classifier and save it."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")

    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[ml_pipeline.model] Saved model to {model_path}")

    metrics = {"accuracy": float(acc)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ml_pipeline.model] Saved metrics to {metrics_path}")
    
    metadata = {
        "model_version": model_version,
        "dataset": "breast_cancer",
        "model_type": "logistic_regression",
        "accuracy": float(acc)
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[ml_pipeline.model] Saved metadata to {metadata_path}")

    return acc

def promote_model(
    bucket_name: str = "seis765-finalproject-560852306721-us-east-1-an", 
    model_path: str = "models/breast_cancer_model.pkl",
    metrics_path: str = "eval/breast_cancer_metrics.json",
    metadata_path: str = "models/breast_cancer_metadata.json",
    accuracy_threshold: float = 0.94,
) -> str:
    """A model should only be published to s3 if it meets a quality threshold."""

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    accuracy = metrics["accuracy"]

    if accuracy < accuracy_threshold:
        raise ValueError(
            f"Model accuracy {accuracy:.4f} is below threshold {accuracy_threshold:.2f}. Promotion failed."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_version = metadata["model_version"]

    s3_prefix = f"models/{model_version}"
    s3 = boto3.client("s3")

    s3.upload_file(model_path, bucket_name, f"{s3_prefix}/model.pkl")
    s3.upload_file(metrics_path, bucket_name, f"{s3_prefix}/metrics.json")
    s3.upload_file(metadata_path, bucket_name, f"{s3_prefix}/metadata.json")

    print(f"[ml_pipeline.bc_model] Promoted model to s3://{bucket_name}/{s3_prefix}/")

    return f"s3://{bucket_name}/{s3_prefix}/"