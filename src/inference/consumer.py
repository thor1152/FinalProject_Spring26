import boto3
import logging
from src.inference.settings import settings
from src.inference.handler import InferenceHandler

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class SQSConsumer:
    def __init__(self):
        self.sqs = boto3.client("sqs", region_name=settings.region_name)
        self.handler = InferenceHandler()

    def start(self):
        logger.info(f"Starting consumer for queue: {settings.queue_url}")
        
        while True:
            response = self.sqs.receive_message(
                QueueUrl=settings.queue_url,
                MaxNumberOfMessages=settings.max_messages,
                WaitTimeSeconds=settings.poll_wait_time
            )

            messages = response.get("Messages", [])
            for msg in messages:
                success = self.handler.handle(msg["Body"])
                
                if success:
                    self.sqs.delete_message(
                        QueueUrl=settings.queue_url,
                        ReceiptHandle=msg["ReceiptHandle"]
                    )
                    logger.debug(f"Message {msg['MessageId']} deleted from queue.")

if __name__ == "__main__":
    consumer = SQSConsumer()
    consumer.start()