# src/app/bc_api.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# Explicit request schema for Breast Cancer dataset (30 features)
class BreastCancerRequest(BaseModel):
   model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189
            }
        }
    )
   mean_radius: float
   mean_texture: float
   mean_perimeter: float
   mean_area: float
   mean_smoothness: float
   mean_compactness: float
   mean_concavity: float
   mean_concave_points: float
   mean_symmetry: float
   mean_fractal_dimension: float

   radius_error: float
   texture_error: float
   perimeter_error: float
   area_error: float
   smoothness_error: float
   compactness_error: float
   concavity_error: float
   concave_points_error: float
   symmetry_error: float
   fractal_dimension_error: float

   worst_radius: float
   worst_texture: float
   worst_perimeter: float
   worst_area: float
   worst_smoothness: float
   worst_compactness: float
   worst_concavity: float
   worst_concave_points: float
   worst_symmetry: float
   worst_fractal_dimension: float

def create_app(
    model_path: str = "models/breast_cancer_model.pkl",
    metadata_path: str = "models/breast_cancer_metadata.json"
):
    """
    Creates a FastAPI app that serves predictions for the Breast Cancer model
    and exposes model metadata.

    Example values from a typical record:
      - target 0 = malignant
      - target 1 = benign
    """
    # Helpful guard so students get a clear error if they forgot to train first
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the DAG or scripts/train_model.py)."
        )

    if not Path(metadata_path).exists():
        raise RuntimeError(
            f"Metadata file not found at '{metadata_path}'. "
            "Train the model first so metadata.json is created."
        )

    model = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    app = FastAPI(title="Breast Cancer Model API")

    # Map numeric predictions to class names
    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {
            "message": "Breast Cancer model is ready for inference!",
            "classes": target_names,
        }

    @app.get("/model/info")
    def model_info():
        return metadata

    @app.post("/predict")
    def predict(request: BreastCancerRequest_bc):
        # Convert request into the correct shape (1 x 30)
        X = pd.DataFrame([{
            "mean_radius": request.mean_radius,
            "mean_texture": request.mean_texture,
            "mean_perimeter": request.mean_perimeter,
            "mean_area": request.mean_area,
            "mean_smoothness": request.mean_smoothness,
            "mean_compactness": request.mean_compactness,
            "mean_concavity": request.mean_concavity,
            "mean_concave_points": request.mean_concave_points,
            "mean_symmetry": request.mean_symmetry,
            "mean_fractal_dimension": request.mean_fractal_dimension,
        
            "radius_error": request.radius_error,
            "texture_error": request.texture_error,
            "perimeter_error": request.perimeter_error,
            "area_error": request.area_error,
            "smoothness_error": request.smoothness_error,
            "compactness_error": request.compactness_error,
            "concavity_error": request.concavity_error,
            "concave_points_error": request.concave_points_error,
            "symmetry_error": request.symmetry_error,
            "fractal_dimension_error": request.fractal_dimension_error,
        
            "worst_radius": request.worst_radius,
            "worst_texture": request.worst_texture,
            "worst_perimeter": request.worst_perimeter,
            "worst_area": request.worst_area,
            "worst_smoothness": request.worst_smoothness,
            "worst_compactness": request.worst_compactness,
            "worst_concavity": request.worst_concavity,
            "worst_concave_points": request.worst_concave_points,
            "worst_symmetry": request.worst_symmetry,
            "worst_fractal_dimension": request.worst_fractal_dimension,
        }])
        
        X = X.rename(columns={
            "mean_radius": "mean radius",
            "mean_texture": "mean texture",
            "mean_perimeter": "mean perimeter",
            "mean_area": "mean area",
            "mean_smoothness": "mean smoothness",
            "mean_compactness": "mean compactness",
            "mean_concavity": "mean concavity",
            "mean_concave_points": "mean concave points",
            "mean_symmetry": "mean symmetry",
            "mean_fractal_dimension": "mean fractal dimension",
    
            "radius_error": "radius error",
            "texture_error": "texture error",
            "perimeter_error": "perimeter error",
            "area_error": "area error",
            "smoothness_error": "smoothness error",
            "compactness_error": "compactness error",
            "concavity_error": "concavity error",
            "concave_points_error": "concave points error",
            "symmetry_error": "symmetry error",
            "fractal_dimension_error": "fractal dimension error",
    
            "worst_radius": "worst radius",
            "worst_texture": "worst texture",
            "worst_perimeter": "worst perimeter",
            "worst_area": "worst area",
            "worst_smoothness": "worst smoothness",
            "worst_compactness": "worst compactness",
            "worst_concavity": "worst concavity",
            "worst_concave_points": "worst concave points",
            "worst_symmetry": "worst symmetry",
            "worst_fractal_dimension": "worst fractal dimension",
    })


        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))

        return {"prediction": target_names[idx], "class_index": idx}

    # return the FastAPI app
    return app