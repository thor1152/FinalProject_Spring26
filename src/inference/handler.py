import json
import boto3
import joblib
import logging
import pandas as pd
from datetime import datetime, timezone
from src.inference.settings import settings
from src.inference.schema import InferenceMessage, InferenceResult

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

class InferenceHandler:
    def __init__(self):
        self.s3_client = boto3.client("s3", region_name=settings.region_name)
        self.model = self._load_model()

    def _load_model(self):
        local_path = "/tmp/model.pkl"
        logger.info(f"Downloading model from s3://{settings.bucket_name}/{settings.model_key}")
        self.s3_client.download_file(settings.bucket_name, settings.model_key, local_path)
        logger.info("Model loaded successfully.")
        return joblib.load(local_path)

    def handle(self, message_body: str) -> bool:
        try:
            data = json.loads(message_body)
            msg = InferenceMessage(**data)

            df = pd.DataFrame([msg.features], columns=FEATURE_NAMES)
            pred = int(self.model.predict(df)[0])

            result = InferenceResult(
                batch_id=msg.batch_id,
                record_id=msg.record_id,
                prediction=pred,
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

            result_key = f"predictions/{msg.batch_id}_{msg.record_id}.json"
            self.s3_client.put_object(
                Bucket=settings.bucket_name,
                Key=result_key,
                Body=result.model_dump_json(indent=2)
            )
            logger.info(f"Processed {msg.record_id} -> Prediction: {pred}")
            return True

        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return False
