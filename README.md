# GastroAI local models

This repository contains the web application and local inference services. Supabase is used for authentication and saved history only; it is not a model-training data source.

## Setup

1. Install Node.js and Python 3.11 or newer.
2. From this directory, install the web dependencies with `npm install`.
3. Copy `.env.example` to `.env` and fill in your own Supabase project URL and anon key. Do not commit `.env`.
4. Install the base model dependencies and train the clinical/Kvasir classifiers:

   ```cmd
   python -m pip install -r requirements-ml.txt
   npm run train:local
   ```

5. For YOLOv7, use the separate Python 3.11 environment and train from the local `..\data\yolo_polyp` dataset:

   ```cmd
   npm run setup:yolov7
   npm run train:yolov7
   ```

   The training script clones the official [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) source into an ignored local directory, fine-tunes YOLOv7-tiny from its official pretrained weights, and evaluates on the dataset's held-out `test` split. It defaults to five epochs for a first CPU run; request a longer training run with `npm run train:yolov7 -- --epochs 50`. No CUDA GPU was detected during setup, so longer runs may take hours. The dataset and trained weights remain local and are not included in this repository.

6. Start each local service in its own terminal:

   ```cmd
   npm run models:local
   ```

   ```cmd
   npm run models:yolov7
   ```

7. In another terminal, start the web app:

   ```cmd
   npm run dev
   ```

The Kvasir model classifies endoscopy images into eight dataset categories. The YOLOv7 model detects and localizes polyp-shaped regions using bounding boxes. Neither model diagnoses cancer or replaces clinical review.

## Local/private files

Datasets, model artifacts and weights, Python environments, Node dependencies, `.env` files, and editor-specific `.bolt` data are excluded from Git. Do not add patient records, credentials, or private data to the repository.
