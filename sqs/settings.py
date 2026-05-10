import os

QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/560852306721/MLOps-FinalProject-Spring26"
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "3"))
VISIBILITY_TIMEOUT = int(os.getenv("VISIBILITY_TIMEOUT", "30"))
WAIT_TIME = int(os.getenv("WAIT_TIME", "20"))