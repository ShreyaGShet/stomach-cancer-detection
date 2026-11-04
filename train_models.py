import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR.parent / "data"
ARTIFACT_DIR = PROJECT_DIR / "model_artifacts"
CLINICAL_NUMERIC = ["age", "family_history", "smoking", "alcohol", "h_pylori"]
CLINICAL_CATEGORICAL = ["gender"]
IMAGE_CLASSES_DIR = DATA_DIR / "kvasir" / "kvasir-dataset-v2"


def extract_image_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(
        image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    features: list[float] = []

    for channel in range(3):
        values = rgb[:, :, channel]
        features.extend([float(values.mean()), float(values.std())])
        features.extend(np.histogram(values, bins=16, range=(0, 1), density=True)[0].tolist())

    for row in range(4):
        for column in range(4):
            patch = rgb[row * 16:(row + 1) * 16, column * 16:(column + 1) * 16]
            for channel in range(3):
                values = patch[:, :, channel]
                features.extend([float(values.mean()), float(values.std())])

    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    features.extend(np.histogram(gray, bins=16, range=(0, 1), density=True)[0].tolist())
    features.extend([float(gray.mean()), float(gray.std())])
    return np.asarray(features, dtype=np.float32)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classes": [str(value) for value in classes.tolist()],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }
    if len(classes) == 2:
        positive_index = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, positive_index]))
    else:
        metrics["auc"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro", labels=classes)
        )
    return metrics


def train_clinical_model() -> tuple[Any, dict[str, Any]]:
    csv_path = DATA_DIR / "clinical" / "gastric_cancer_detection_dataset.csv"
    frame = pd.read_csv(
        csv_path,
        usecols=[
            "age",
            "gender",
            "family_history",
            "smoking_habits",
            "alcohol_consumption",
            "helicobacter_pylori_infection",
            "label",
        ],
    ).rename(
        columns={
            "smoking_habits": "smoking",
            "alcohol_consumption": "alcohol",
            "helicobacter_pylori_infection": "h_pylori",
        }
    )
    for column in CLINICAL_NUMERIC + ["label"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["gender"] = frame["gender"].fillna("unknown").astype(str).str.lower()
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)

    x = frame[CLINICAL_NUMERIC + CLINICAL_CATEGORICAL]
    y = frame["label"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), CLINICAL_NUMERIC),
            ("gender", OneHotEncoder(handle_unknown="ignore"), CLINICAL_CATEGORICAL),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test, predictions, probabilities, model.classes_)
    metrics.update(
        {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "features": CLINICAL_NUMERIC + CLINICAL_CATEGORICAL,
            "source": "data/clinical/gastric_cancer_detection_dataset.csv",
        }
    )
    return model, metrics


def train_image_model() -> tuple[Any, dict[str, Any]]:
    split_path = DATA_DIR / "splits.csv"
    with split_path.open(newline="", encoding="utf-8-sig") as file:
        split_rows = list(csv.DictReader(file))

    images: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    splits: dict[str, str] = {}
    for row in split_rows:
        split = row["split"]
        if split not in {"train", "test"}:
            continue
        label = row["label"]
        filename = Path(row["path"].replace("\\", "/")).name
        image_path = IMAGE_CLASSES_DIR / label / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in data/splits.csv was not found: {image_path}")
        key = f"{label}/{filename}"
        with Image.open(image_path) as image:
            images[key] = extract_image_features(image)
        labels[key] = label
        splits[key] = split

    train_keys = [key for key, split in splits.items() if split == "train"]
    test_keys = [key for key, split in splits.items() if split == "test"]
    if not train_keys or not test_keys:
        raise ValueError("data/splits.csv must contain non-empty train and test image splits.")

    x_train = np.stack([images[key] for key in train_keys])
    y_train = np.asarray([labels[key] for key in train_keys])
    x_test = np.stack([images[key] for key in test_keys])
    y_test = np.asarray([labels[key] for key in test_keys])
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test, predictions, probabilities, model.classes_)
    metrics.update(
        {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "source": "data/kvasir/kvasir-dataset-v2 + data/splits.csv",
            "class_counts": {
                str(label): int(sum(value == label for value in labels.values()))
                for label in model.classes_.tolist()
            },
        }
    )
    return model, metrics


def save_artifact(name: str, value: Any) -> None:
    temporary_path = ARTIFACT_DIR / f"{name}.joblib.tmp"
    joblib.dump(value, temporary_path)
    temporary_path.replace(ARTIFACT_DIR / f"{name}.joblib")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    clinical_model, clinical_metrics = train_clinical_model()
    image_model, image_metrics = train_image_model()
    save_artifact("clinical_model", clinical_model)
    save_artifact("image_model", image_model)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "clinical": clinical_metrics,
        "image": image_metrics,
    }
    metrics_path = ARTIFACT_DIR / "metrics.json"
    temporary_metrics_path = metrics_path.with_suffix(".json.tmp")
    temporary_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temporary_metrics_path.replace(metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
