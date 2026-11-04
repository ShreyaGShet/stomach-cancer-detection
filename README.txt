The local training pipeline uses gastric_cancer_detection_dataset.csv.

From the project directory:
  1. Install the Python dependencies: python -m pip install -r requirements-ml.txt
  2. Train both models: npm run train:local
  3. Start the local prediction API in a terminal: npm run models:local
  4. Start the web app in another terminal: npm run dev

The clinical model deliberately uses only age, gender, family history, smoking,
alcohol consumption, and H. pylori status. Endoscopy, biopsy, CT, and molecular
target/result columns are excluded to avoid label leakage. The remaining
clinical form values are saved with analyses but are not model inputs because
they are not present in this dataset. The image classifier uses the train and
test assignments in data/splits.csv; the validation split is not used for
training or reported test metrics.

The YOLOv7 polyp detector uses the YOLO-format images and labels under
data/yolo_polyp. From the project directory, run npm run setup:yolov7 and
npm run train:yolov7. This creates an isolated Python 3.11 environment,
fine-tunes YOLOv7-tiny for five epochs by default, and reports precision,
recall, mAP@0.5, and mAP@0.5:0.95 on the held-out test split. Longer runs can
be requested with npm run train:yolov7 -- --epochs 50. Start its separate API
with npm run models:yolov7. Data, pretrained weights, and trained model
artifacts are local-only and are not published to GitHub.
