from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.producer import populate_sqs_queue

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="populate_inference_queue",
    default_args=default_args,
    description="Reads test dataset and pushes individual inference jobs to SQS",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def populate_queue_wrapper():
        # IMPORTANT: Replace the queue_url with your actual SQS URL from the AWS Console
        return populate_sqs_queue(
            data_path="data/breast_cancer.csv",
            queue_url="https://sqs.us-east-1.amazonaws.com/560852306721/MLOps-FinalProject-Spring26" 
        )

    populate_task = PythonOperator(
        task_id="send_messages_to_sqs",
        python_callable=populate_queue_wrapper,
    )


