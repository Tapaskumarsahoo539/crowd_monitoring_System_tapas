# AI-Based Smart Crowd Monitoring and Analysis System using YOLOv8

This project is a complete AI-based Smart Crowd Monitoring and Analysis System built using YOLOv8 and Deep Learning. It detects, monitors, analyzes, and counts people in crowded public areas in real-time.

## Project Objectives
1. **Detect people** in crowded environments using YOLOv8.
2. **Count** the number of people automatically.
3. Monitor **crowd density** in real-time.
4. Detect **overcrowding** situations automatically (e.g., alert if count > threshold).
5. Support **image, video, and live webcam** input.

## Technologies Used
- **Python 3.x**
- **YOLOv8 (Ultralytics)**
- **OpenCV** (for image/video processing)
- **NumPy & SciPy** (for array and dataset `.mat` operations)
- **Matplotlib** (for visualization)
- **Google Colab** (for GPU training)

## Project Structure
```text
crowd_monitoring_System_tapas/
│
├── dataset/                    # Dataset folder
│   ├── images/                 # Original UCF-QNRF images
│   ├── ground_truth/           # Original MAT files
│   └── yolo_format/            # Converted dataset for YOLO
│
├── models/                     # Saved YOLOv8 models (e.g., yolov8n.pt, best.pt)
├── outputs/                    # Directory for processed image/video outputs
├── notebooks/                  # Google Colab notebooks
├── scripts/                    # Core Python scripts
│   ├── 1_preprocess_ucf_qnrf.py
│   ├── 2_train_yolov8.py
│   ├── 3_inference_yolov8.py
│   ├── 3_inference_yolo11.py
│   └── 3_inference_yolo12.py
├── data.yaml                   # YOLO dataset configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Step-by-Step Setup Guide

### 1. Install Requirements
Ensure you have Python installed. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation (UCF-QNRF)
1. Download the UCF-QNRF dataset from Kaggle: [faihajalamtopu/ucf-qnrf](https://www.kaggle.com/datasets/faihajalamtopu/ucf-qnrf).
2. Extract the dataset and place the `.jpg` images into `dataset/images/` and the `.mat` annotation files into `dataset/ground_truth/`.
3. Run the preprocessing script to convert point annotations to YOLO bounding box format:
```bash
python scripts/1_preprocess_ucf_qnrf.py
```
This will populate the `dataset/yolo_format/` directory with `images` and `labels`.

### 3. Training the Model
To train the YOLOv8 model on the preprocessed UCF-QNRF dataset, run:
```bash
python scripts/2_train_yolov8.py
```
This script uses `yolov8n.pt` as the base model and trains it using the `data.yaml` configuration. The best weights will be saved automatically by Ultralytics in `runs/train/crowd_model/weights/best.pt`.

### 4. Real-time Inference and Crowd Monitoring
You can use the inference script to test the model on images, videos, or your webcam.

**To run on a static image:**
```bash
python scripts/3_inference_yolov8.py --source "path/to/image.jpg" --model "yolov8n.pt" --threshold 20
```

**To run on a video file:**
```bash
python scripts/3_inference_yolov8.py --source "path/to/video.mp4" --model "yolov8n.pt" --threshold 20
```

**To run using your live webcam:**
```bash
python scripts/3_inference_yolov8.py --source 0 --model "yolov8n.pt" --threshold 20
```
*(Press 'q' to quit the webcam view)*

> **Note:** If you have trained your own model on the UCF-QNRF dataset, pass its path to `--model`, e.g., `--model ../runs/train/crowd_model/weights/best.pt`. Otherwise, passing `yolov8n.pt` will automatically download and use the official pre-trained weights which can natively detect people.

### 5. Google Colab (Recommended for Training)
If you don't have a local GPU, we highly recommend using the provided Jupyter Notebook.
1. Upload `notebooks/Crowd_Monitoring_Colab.ipynb` to Google Colab.
2. Follow the step-by-step instructions in the notebook to download the dataset, train the model, and run inference using a free T4 GPU.

---

### 6. Using YOLO11 (Alternative)
We also provide support for the newer YOLO11 models. To use YOLO11 instead of YOLOv8 without modifying the base code:

**Training with YOLO11:**
Run the dedicated YOLO11 training script:
```bash
python scripts/2_train_yolo11.py
```

**Inference with YOLO11:**
You can run the existing inference script but pass a YOLO11 model weights file:
```bash
python scripts/3_inference_yolo11.py --source 0 --model "yolo11m.pt" --threshold 20
```
> **Note:** Make sure to upgrade your ultralytics package (`pip install -U ultralytics`) to use YOLO11 models.

**Google Colab (YOLO11):**
There is a dedicated Google Colab notebook for YOLO11 training and inference located at `notebooks/Crowd_Monitoring_YOLO11_Colab.ipynb`.

---

### 7. Using YOLO12 (Alternative)
We also provide support for the newest YOLO12 models. To use YOLO12 instead of YOLOv8 without modifying the base code:

**Training with YOLO12:**
Run the dedicated YOLO12 training script:
```bash
python scripts/2_train_yolo12.py
```

**Inference with YOLO12:**
You can run the existing inference script but pass a YOLO12 model weights file:
```bash
python scripts/3_inference_yolo12.py --source 0 --model "yolo12m.pt" --threshold 20
```
> **Note:** Make sure to upgrade your ultralytics package (`pip install -U ultralytics`) to use YOLO12 models.

**Google Colab (YOLO12):**
There is a dedicated Google Colab notebook for YOLO12 training and inference located at `notebooks/Crowd_Monitoring_YOLO12_Colab.ipynb`.

---
**Developed for B.Tech CSE (AIML) Final Year Project.**
