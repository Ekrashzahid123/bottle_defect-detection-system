# 🍾 Industrial Bottle Defect Detection System (100% Pure TensorFlow 2.x)

A production-ready, leak-free, and defensible Computer Vision system to detect **Normal** vs. **Defective** bottles on a manufacturing production line. Built entirely in **TensorFlow 2.x / Keras**, **MobileNetV3**, **FastAPI**, and **Streamlit**.

---

## 📋 Table of Contents
1. [Core Architectural Decisions & Integrity Fixes](#-core-architectural-decisions--integrity-fixes)
2. [Model Selection & Rationale](#-model-selection--rationale)
3. [Leak-Free Group-Stratified Dataset Pipeline](#-leak-free-group-stratified-dataset-pipeline)
4. [Project Structure](#-project-structure)
5. [Quick Start & Setup Instructions](#-quick-start--setup-instructions)
6. [Training in TensorFlow](#-training-in-tensorflow)
7. [Evaluation & Confusion Matrix Analysis](#-evaluation--confusion-matrix-analysis)
8. [FastAPI Production Service](#-fastapi-production-service)
9. [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
10. [Docker Deployment](#-docker-deployment)
11. [Interview Defense Cheat-Sheet](#-interview-defense-cheat-sheet)

---

## 🛡️ Core Architectural Decisions & Integrity Fixes

To ensure full scientific and engineering defensibility:

| Critical Factor | Problem in Naive Implementations | How We Solved It in TensorFlow |
| :--- | :--- | :--- |
| **Data Leakage (Duplicate Images)** | Roboflow creates 3 augmented copies per physical image. Random splitting puts copies of the same bottle across train & test, yielding inflated "100%" test accuracy. | **Group-Aware Stratified Splitting (`GroupShuffleSplit`)**: All augmented views of a physical bottle (`base_id`) are strictly grouped into the **same** split. Zero overlap between Train, Val, and Test. |
| **Label Source Integrity** | Guessing labels from filename prefixes (`IMG_11xx` vs `IMG_14xx`). | **Pure YOLO Annotation Parsing**: Labels are parsed 100% from YOLO `.txt` files. No heuristic filename guesswork. |
| **Multi-Object Handling** | Taking only the first token or crashing on multiple boxes. | **Explicit Business Logic**: If **any** object is marked Defective (`class 1`), the product is marked **Defective**. If all are Good (`class 0`), it is **Normal**. |
| **Validation Set Size** | Tiny validation sets (e.g. 20-40 images) leading to high variance. | Full 3-way split: **Train (70%)**, **Validation (15%)**, and **Holdout Test (15%)** guaranteeing statistical significance. |
| **Edge Hardware Compatibility** | Heavy CNNs (ResNet, VGG) requiring GPU. | **MobileNetV3 in TensorFlow**: ~2.5M parameters, Hard-Swish activations, Squeeze-and-Excitation attention, ~10ms CPU inference. |

---

## 🧠 Model Selection & Rationale

```mermaid
flowchart LR
    A[Input 224x224x3] --> B[tf.keras MobileNetV3 Pretrained Backbone]
    B --> C[GlobalAveragePooling2D]
    C --> D[Dropout 0.25]
    D --> E[Dense 2 Units + Softmax]
    E --> F[Normal / Defective Probabilities]
```

- **Backbone**: `tf.keras.applications.MobileNetV3Small` (ImageNet pretrained).
- **Activations**: Hard-Swish $\frac{x \cdot \text{ReLU6}(x+3)}{6}$, avoiding slow exponential operations on embedded CPUs.
- **Attention**: Squeeze-and-Excitation (SE) channel-wise recalibration focusing on bottle-cap closures.
- **Loss**: `SparseCategoricalCrossentropy` with automated inverse-frequency `class_weight`.

---

## 📁 Project Structure

```
bottle_detect detection system/
├── dataset/                      # Raw dataset (train, valid, test images & labels)
├── models/                       # Checkpoints & artifacts
│   ├── best_model.keras          # Saved Keras MobileNetV3 model
│   ├── split_manifest.json       # Leak-free split manifest
│   ├── training_curves.png       # Loss & Accuracy learning curves
│   └── training_metrics.json     # Epoch log
├── metrics/                      # Confusion matrices & evaluation JSON
│   ├── test_confusion_matrix.png
│   ├── valid_confusion_matrix.png
│   └── test_evaluation_report.json
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Group-aware leak-free dataset loader & augmentations
│   ├── model.py                  # Pure TensorFlow MobileNetV3 architecture
│   ├── train.py                  # TensorFlow training pipeline
│   ├── evaluate.py               # Confusion matrix & error analysis
│   └── predictor.py              # Pure TensorFlow inference engine
├── api/
│   ├── __init__.py
│   └── app.py                    # FastAPI service (/health, /predict)
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── Dockerfile                    # Container deployment configuration
├── requirements.txt              # TensorFlow dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quick Start & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Training in TensorFlow

To train the MobileNetV3 model with leak-free group splitting:

```bash
python src/train.py --epochs 12 --batch_size 32 --lr 0.001 --variant small
```

**Training Outputs:**
- `models/best_model.keras`: Checkpoint with highest validation accuracy.
- `models/split_manifest.json`: Exact filepaths and ground-truth labels for Train, Val, and Test sets.
- `models/training_curves.png`: Training vs. Validation Loss and Accuracy curves.

---

## 📈 Evaluation & Confusion Matrix Analysis

To evaluate on the **Holdout Test Set** or **Validation Set**:

```bash
# Evaluate on Holdout Test Set
python src/evaluate.py --split test

# Evaluate on Validation Set
python src/evaluate.py --split valid
```

**Outputs:**
- `metrics/test_confusion_matrix.png`: High-resolution Confusion Matrix plot.
- `metrics/test_evaluation_report.json`: Precision, Recall, F1-Score, ROC-AUC, and detailed False Positive / False Negative breakdown.

---

## ⚡ FastAPI Production Service

Launch the FastAPI backend:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Probe**: `GET /health`
- **Predict Endpoint**: `POST /predict`

### Example `curl` Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dataset/valid/images/IMG_1116_jpg.rf.51ce82d477a8f07b0068121e119bdb8d.jpg"
```

---

## 🖥 Streamlit Interactive Dashboard

Launch the inspection UI:

```bash
streamlit run ui/streamlit_app.py --server.port 8503
```

- **Live Single Inspector**: Upload or pick sample bottle images with real-time status banners and latency timer.
- **Leak-Free Test Evaluation**: 1-click batch evaluation over all holdout test bottles.
- **Model Analytics**: Confusion matrix plots, loss/accuracy curves, and error analysis inspector.

---

## 🐳 Docker Deployment

```bash
docker build -t bottle-defect-detector-tf:latest .
docker run -p 8000:8000 bottle-defect-detector-tf:latest
```

---

## 🎓 Interview Defense Cheat-Sheet

### Q1: How did you ensure there is no data leakage across splits?
> **Answer**: *"Roboflow creates 3 augmented variations per base image. If you use a random train/test split, augmented copies of the same bottle end up in both train and test, which artifically inflates performance. We solved this by implementing `GroupShuffleSplit` on the base image ID (`IMG_XXXX`), guaranteeing that all augmentations of a physical bottle stay strictly inside the same partition."*

### Q2: How did you handle labeling and multi-object edge cases?
> **Answer**: *"We do not rely on filename heuristics. All labels are parsed directly from the YOLO annotation text files. If an image contains multiple objects, we apply industrial QA logic: if **any** bounding box contains class `1` (open cap / defect), the entire product is classified as Defective. If all boxes are class `0`, it is classified as Normal."*

### Q3: Why is MobileNetV3 in TensorFlow ideal for this factory deployment?
> **Answer**: *"MobileNetV3 uses depthwise separable convolutions, Hard-Swish activations, and Squeeze-and-Excitation attention to deliver high accuracy with only ~2.5M parameters. In TensorFlow, it converts directly to TFLite for deployment onto edge hardware (e.g. Raspberry Pi / Jetson) with sub-15ms inference per bottle."*
