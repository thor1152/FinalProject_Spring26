from pydantic import BaseModel, Field, field_validator
from typing import List

class InferenceMessage(BaseModel):
    batch_id: str = Field(..., description="Unique ID for the inference batch")
    record_id: str = Field(..., description="Unique ID for the record")
    features: List[float] = Field(..., description="List of feature values")

    @field_validator('features')
    @classmethod
    def validate_feature_count(cls, v):
        if len(v) != 30:
            raise ValueError(f"Breast cancer model requires exactly 30 features, got {len(v)}")
        return v

class InferenceResult(BaseModel):
    batch_id: str
    record_id: str
    prediction: int
    timestamp: str
