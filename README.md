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
│   └── 3_inference.py
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
python scripts/3_inference.py --source "path/to/image.jpg" --model "yolov8n.pt" --threshold 20
```

**To run on a video file:**
```bash
python scripts/3_inference.py --source "path/to/video.mp4" --model "yolov8n.pt" --threshold 20
```

**To run using your live webcam:**
```bash
python scripts/3_inference.py --source 0 --model "yolov8n.pt" --threshold 20
```
*(Press 'q' to quit the webcam view)*

> **Note:** If you have trained your own model on the UCF-QNRF dataset, pass its path to `--model`, e.g., `--model ../runs/train/crowd_model/weights/best.pt`. Otherwise, passing `yolov8n.pt` will automatically download and use the official pre-trained weights which can natively detect people.

### 5. Google Colab (Recommended for Training)
If you don't have a local GPU, we highly recommend using the provided Jupyter Notebook.
1. Upload `notebooks/Crowd_Monitoring_Colab.ipynb` to Google Colab.
2. Follow the step-by-step instructions in the notebook to download the dataset, train the model, and run inference using a free T4 GPU.

---
**Developed for B.Tech CSE (AIML) Final Year Project.**
