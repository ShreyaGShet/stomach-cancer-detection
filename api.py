import json
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from ml.train_models import ARTIFACT_DIR, CLINICAL_CATEGORICAL, CLINICAL_NUMERIC, extract_image_features

app = FastAPI(title="Local Dataset Models")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ClinicalRequest(BaseModel):
    age: float = Field(ge=0, le=120)
    gender: str
    family_history: bool
    alcohol_consumption: str
    smoking_status: str
    helicobacter_pylori: bool


def load_model(name: str) -> Any:
    artifact = ARTIFACT_DIR / f"{name}.joblib"
    if not artifact.is_file():
        raise HTTPException(
            status_code=503,
            detail="Local model artifacts are missing. Run `npm run train:local` from the project directory.",
        )
    return joblib.load(artifact)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Training metrics are unavailable. Run `npm run train:local` from the project directory.",
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/clinical/predict")
def predict_clinical(request: ClinicalRequest) -> dict[str, Any]:
    row = pd.DataFrame(
        [
            {
                "age": request.age,
                "family_history": int(request.family_history),
                "smoking": int(request.smoking_status.lower() != "never"),
                "alcohol": int(request.alcohol_consumption.lower() not in {"none", "light"}),
                "h_pylori": int(request.helicobacter_pylori),
                "gender": request.gender.lower(),
            }
        ],
        columns=CLINICAL_NUMERIC + CLINICAL_CATEGORICAL,
    )
    model = load_model("clinical_model")
    probabilities = model.predict_proba(row)[0]
    classes = model.classes_
    positive_index = int(np.where(classes == 1)[0][0]) if 1 in classes else -1
    malignant_probability = float(probabilities[positive_index]) if positive_index >= 0 else 0.0
    prediction = "malignant" if malignant_probability >= 0.5 else "benign"
    confidence = malignant_probability if prediction == "malignant" else 1 - malignant_probability
    return {
        "prediction_result": prediction,
        "prediction_confidence": confidence,
        "risk_score": round(malignant_probability * 100),
    }


@app.post("/image/predict")
async def predict_image(image: UploadFile = File(...)) -> dict[str, Any]:
    contents = await image.read(15 * 1024 * 1024 + 1)
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image file must be 15 MB or smaller.")
    try:
        with Image.open(BytesIO(contents)) as uploaded_image:
            features = extract_image_features(uploaded_image)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from error

    model = load_model("image_model")
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    classes = model.classes_
    best_index = int(np.argmax(probabilities))
    results = [
        {"class": str(label), "confidence": float(probability)}
        for label, probability in sorted(
            zip(classes.tolist(), probabilities.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "topClass": str(classes[best_index]),
        "confidence": float(probabilities[best_index]),
        "risk_level": "not_assessed",
        "results": results,
    }
