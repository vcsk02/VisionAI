# VisionAI – CRISP-DM Process

This document describes our project using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. VisionAI is an end-to-end computer vision system that trains an image model and serves its predictions through a web UI.

---

## 1. Business Understanding

### 1.1 Problem Statement

Modern computer vision models are powerful but hard to experiment with for non-experts. Our goal is to:

- Train a performant vision model on a labeled image dataset.
- Expose that model through a simple web interface (VisionAI) for inference and inspection.
- Provide rich visualizations of model metrics and predictions.

Concretely, we tackle the task of **[image classification / object detection / segmentation – choose one]** on the **[DATASET_NAME]** dataset.

### 1.2 Objectives & Success Criteria

- **Predictive performance:**  
  - Target test accuracy ≥ **[X%]** (or other metric such as F1, mAP, IoU).
- **Usability:**  
  - A web UI where a user can upload an image and see model predictions and confidence scores.
- **Reproducibility:**  
  - A Colab / Jupyter notebook that can be re-run to retrain the model from scratch.
- **Explainability & evaluation:**  
  - Visualizations for loss/accuracy curves, confusion matrix, and qualitative examples.

---

## 2. Data Understanding

### 2.1 Data Collection

- **Source:**  
  - **[Describe dataset: e.g., CIFAR-10 / custom dataset / Kaggle dataset]**
- **Size:**  
  - **[N]** images across **[K]** classes.
- **Format:**  
  - RGB images, originally sized **[H×W]**, stored as **[e.g., PNG/JPEG]**.

### 2.2 Data Description

- Each sample consists of:
  - An input image.
  - A categorical label from the set: **[list class names or examples]**.
- Class distribution:
  - **[Brief note on whether classes are balanced or imbalanced]**.

### 2.3 Data Quality

During initial exploration we checked for:

- Missing or corrupted files.
- Incorrect labels or obvious annotation errors.
- Class imbalance issues.

Any problematic samples were **[removed / fixed / kept but flagged]** and this is documented in the training notebook.

---

## 3. Data Preparation

### 3.1 Train/Validation/Test Split

We split the dataset into:

- **Train:** ~**[X%]** of the data  
- **Validation:** ~**[Y%]**  
- **Test:** ~**[Z%]**

Splits are stratified by class where possible to maintain class distribution.

### 3.2 Preprocessing

For all images we apply:

- Resize (if needed) to **[e.g., 224×224]**.
- Conversion to tensor.
- Normalization using dataset-specific mean and standard deviation.

### 3.3 Data Augmentation

To improve generalization and reduce overfitting we apply:

- Random horizontal flip  
- Random crop / random resized crop  
- **[Any other augmentations: color jitter, rotation, etc.]**

In the notebook we explain **why** each augmentation is chosen and show its effect on performance in the ablation section.

---

## 4. Modeling

### 4.1 Model Choice

We use **[e.g., ResNet-18]** as our primary architecture:

- Pretrained on ImageNet and fine-tuned on our dataset.
- Final fully-connected layer adapted to output **K** classes.

We also experiment with **[alternative models, e.g., ResNet-34, EfficientNet, custom CNN]** as part of our ablation studies.

### 4.2 Training Configuration

- **Loss function:** Cross-entropy loss  
  - Suitable for multi-class classification with mutually exclusive labels.
- **Optimizer:** Adam  
  - Adaptive learning rate, good default choice for deep networks.
- **Learning rate:** **[e.g., 1e-3]**
- **Batch size:** **[e.g., 64 or 128]**
- **Epochs:** **[e.g., 20–50]**, with early stopping / model checkpointing based on validation performance.

### 4.3 Hyperparameter Tuning & Ablation Studies

We vary:

- Learning rate (e.g. 1e-2, 1e-3, 1e-4)
- Data augmentation (with vs without)
- Model depth (e.g. ResNet-18 vs ResNet-34)

Results are summarized in the **Experiments and Results** section of our report and in tables inside the notebook.

---

## 5. Evaluation

### 5.1 Metrics

We evaluate on the held-out test set using:

- **Accuracy**
- **Precision, Recall, F1-score** per class (for imbalanced or multi-class data)
- **Confusion matrix** to inspect which classes are commonly confused

These metrics are visualized as:

- Training vs validation loss curves
- Training vs validation accuracy curves
- Confusion matrix heatmap
- **[Optionally ROC / PR curves, etc.]**

### 5.2 Results Summary

- Best model: **[model name and configuration]**
- Test accuracy: **[X%]**  
- Key observations:
  - **[e.g., augmentation improved generalization by Y%.]**
  - **[e.g., deeper model did not help due to overfitting.]**

### 5.3 Error Analysis

We inspect typical failure cases:

- Images with heavy occlusion, clutter, or atypical viewpoints.
- Classes with very few examples.
- Misclassified samples visualized in the notebook to understand model limitations.

---

## 6. Deployment

### 6.1 Model Export

After training, we export the model weights to:

- `exported_model/visionai_model.pth` (PyTorch) **[or your actual path]**

This file is versioned in the repo and can be loaded by the backend.

### 6.2 Inference Service

We expose the model via a backend service (e.g., FastAPI/Flask):

1. Receive image from the VisionAI frontend.
2. Apply the same preprocessing steps as in the notebook.
3. Run inference with the trained model.
4. Return predictions, confidence scores, and any structured outputs (e.g., bounding boxes).

The VisionAI web client calls this API using a configurable `VITE_API_BASE_URL`.

### 6.3 End-to-End System

The full pipeline is:

1. **Training & Evaluation:** Run `notebooks/visionai_model_training.ipynb` to retrain and evaluate the model.
2. **Export:** Save the best model checkpoint.
3. **Deployment:** Start the backend service loading this checkpoint.
4. **Frontend:** Run the VisionAI Vite app; users upload images and see predictions + visualizations.

This end-to-end setup demonstrates the full CRISP-DM cycle from business understanding through deployment.
