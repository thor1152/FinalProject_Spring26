from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.model import promote_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="promote",
    default_args=default_args,
    description="promote models to s3 that exceed accuracy threshold",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def promote_model_wrapper(
        bucket_name: str,
        model_path: str,
        metrics_path: str,
        metadata_path: str,
        accuracy_threshold: float,
    ):
        return promote_model(
            bucket_name=bucket_name,
            model_path=model_path,
            metrics_path=metrics_path,
            metadata_path=metadata_path,
            accuracy_threshold=accuracy_threshold,    
        )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
        op_kwargs={
            "bucket_name": "seis765-finalproject-560852306721-us-east-1-an",
            "model_path": "models/breast_cancer_model.pkl",
            "metrics_path": "eval/breast_cancer_metrics.json",
            "metadata_path": "models/breast_cancer_metadata.json",
            "accuracy_threshold": 0.94,
        },
    )