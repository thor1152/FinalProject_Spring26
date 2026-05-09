from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import generate_data, load_data
from ml_pipeline.model import train_model, promote_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="end_to_end_ml_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # 1. Generate Data Task
    generate_task = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
        op_kwargs={"output_path": "data/breast_cancer.csv"},
    )

    # 2. Train Model Task
    def train_model_wrapper(data_path: str, model_path: str):
        df = load_data(data_path)
        return train_model(df, model_path)

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "model_path": "models/breast_cancer_model.pkl",
        },
    )

    # 3. Promote Model Task
    def promote_model_wrapper():
        return promote_model(
            bucket_name="seis765-finalproject-560852306721-us-east-1-an",
            model_path="models/breast_cancer_model.pkl",
            metrics_path="eval/breast_cancer_metrics.json",
            metadata_path="models/breast_cancer_metadata.json",
            accuracy_threshold=0.94,
        )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
    )

    # Define the order of operations
    generate_task >> train_task >> promote_task
