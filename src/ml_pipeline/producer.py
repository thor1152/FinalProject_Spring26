import json
import boto3
import pandas as pd

def populate_sqs_queue(
    test_data_path: str = "data/test_data.csv", 
    queue_url: str = "https://sqs.us-east-1.amazonaws.com/560852306721/MLOps-FinalProject-Spring26"
) -> int:
    """Reads the saved test data and sends individual records to SQS."""
    
    # 1. Load the pre-split test data
    X_test = pd.read_csv(test_data_path)

    # 2. Initialize the SQS client
    sqs = boto3.client("sqs", region_name="us-east-1")
    messages_sent = 0

    # 3. Iterate and send
    for index, row in X_test.iterrows():
        message_body = {
            "record_id": f"sample_{index:03d}",
            "features": row.tolist()
        }

        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message_body)
        )
        messages_sent += 1

    print(f"[ml_pipeline.producer] Successfully sent {messages_sent} messages to SQS.")
    return messages_sent