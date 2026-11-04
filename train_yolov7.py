import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR.parent / "data" / "yolo_polyp"
ARTIFACT_DIR = PROJECT_DIR / "model_artifacts" / "yolov7"
YOLOV7_DIR = PROJECT_DIR / ".yolov7-upstream"
WEIGHTS_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
PRETRAINED_WEIGHTS = ARTIFACT_DIR / "yolov7-tiny.pt"
TRAINED_WEIGHTS = ARTIFACT_DIR / "polyp_yolov7_tiny.pt"
DATA_YAML = ARTIFACT_DIR / "polyp_dataset.yaml"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def ensure_dataset() -> None:
    required = [
        DATASET_DIR / "images" / split
        for split in ("train", "val", "test")
    ] + [
        DATASET_DIR / "labels" / split
        for split in ("train", "val", "test")
    ]
    missing = [path for path in required if not path.is_dir()]
    if missing:
        missing_paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"The YOLO polyp dataset is incomplete:\n{missing_paths}")


def ensure_yolov7_source() -> None:
    if (YOLOV7_DIR / "train.py").is_file() and (YOLOV7_DIR / "test.py").is_file():
        return
    if YOLOV7_DIR.exists():
        raise FileExistsError(
            f"{YOLOV7_DIR} exists but is not a complete YOLOv7 checkout. "
            "Remove that directory and retry."
        )
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/WongKinYiu/yolov7.git",
            str(YOLOV7_DIR),
        ],
        check=True,
    )


def write_dataset_yaml() -> None:
    config = {
        "train": (DATASET_DIR / "images" / "train").as_posix(),
        "val": (DATASET_DIR / "images" / "val").as_posix(),
        "test": (DATASET_DIR / "images" / "test").as_posix(),
        "nc": 1,
        "names": ["polyp"],
    }
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def download_pretrained_weights() -> None:
    if PRETRAINED_WEIGHTS.is_file():
        return
    PRETRAINED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    request = Request(WEIGHTS_URL, headers={"User-Agent": "local-yolov7-training"})
    with urlopen(request, timeout=60) as response, PRETRAINED_WEIGHTS.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def count_images(split: str) -> int:
    image_dir = DATASET_DIR / "images" / split
    return sum(
        1
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def parse_test_metrics(output: str) -> dict[str, float]:
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 7 and fields[0] == "all":
            try:
                return {
                    "precision": float(fields[3]),
                    "recall": float(fields[4]),
                    "map50": float(fields[5]),
                    "map50_95": float(fields[6]),
                }
            except ValueError:
                continue
    raise RuntimeError("YOLOv7 test output did not contain the expected aggregate metrics row.")


def run_training(epochs: int) -> None:
    train_run_dir = ARTIFACT_DIR / "runs"
    subprocess.run(
        [
            sys.executable,
            "train.py",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--data",
            str(DATA_YAML),
            "--img-size",
            "640",
            "640",
            "--cfg",
            "cfg/training/yolov7-tiny.yaml",
            "--weights",
            str(PRETRAINED_WEIGHTS),
            "--name",
            "polyp-yolov7-tiny",
            "--hyp",
            "data/hyp.scratch.tiny.yaml",
            "--epochs",
            str(epochs),
            "--project",
            str(train_run_dir),
            "--exist-ok",
        ],
        cwd=YOLOV7_DIR,
        check=True,
    )
    run_weights = train_run_dir / "polyp-yolov7-tiny" / "weights" / "best.pt"
    if not run_weights.is_file():
        raise FileNotFoundError(f"YOLOv7 training completed without producing {run_weights}")
    TRAINED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    run_weights.replace(TRAINED_WEIGHTS)


def evaluate_test_split() -> dict[str, float]:
    evaluation_dir = ARTIFACT_DIR / "test_runs"
    process = subprocess.run(
        [
            sys.executable,
            "test.py",
            "--data",
            str(DATA_YAML),
            "--img-size",
            "640",
            "--batch-size",
            "4",
            "--conf-thres",
            "0.001",
            "--iou-thres",
            "0.65",
            "--task",
            "test",
            "--device",
            "cpu",
            "--weights",
            str(TRAINED_WEIGHTS),
            "--project",
            str(evaluation_dir),
            "--name",
            "polyp-test",
            "--exist-ok",
        ],
        cwd=YOLOV7_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    print(process.stdout)
    if process.stderr:
        print(process.stderr, file=sys.stderr)
    return parse_test_metrics(process.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv7-tiny on the local polyp dataset.")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    ensure_dataset()
    ensure_yolov7_source()
    write_dataset_yaml()
    download_pretrained_weights()
    run_training(args.epochs)
    metrics: dict[str, Any] = {
        "model": "YOLOv7-tiny",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs": args.epochs,
        "train_images": count_images("train"),
        "validation_images": count_images("val"),
        "test_images": count_images("test"),
        "source": "data/yolo_polyp",
        "weights": "model_artifacts/yolov7/polyp_yolov7_tiny.pt",
        **evaluate_test_split(),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = METRICS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temporary_path.replace(METRICS_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
