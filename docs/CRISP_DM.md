# VisionAI – CRISP-DM Process

This document describes our project using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. VisionAI is an end-to-end computer vision system that trains an image classification model and serves its predictions through a web UI.

---

## 1. Business Understanding

### 1.1 Problem Statement

Modern computer vision models are powerful but hard to experiment with for non-experts. Our goal is to:

- Train a performant vision model on a labeled image dataset.
- Expose that model through a simple web interface (VisionAI) for inference and inspection.
- Provide rich visualizations of model metrics and predictions.

Concretely, we tackle the task of **multi-class image classification** on a **4-class image dataset** (referred to here as the *VisionAI image dataset*).

### 1.2 Objectives & Success Criteria

- **Predictive performance:**  
  - Achieve test accuracy of at least **80%**, with macro F1-score also around or above **0.80**.
- **Usability:**  
  - Provide a web UI where a user can upload an image and see model predictions and confidence scores in real time.
- **Reproducibility:**  
  - Provide a Colab / Jupyter notebook that can be re-run to retrain the model from scratch and regenerate evaluation artifacts.
- **Explainability & evaluation:**  
  - Provide visualizations for loss/accuracy curves, confusion matrix, classification report, and qualitative examples of failure cases.

---

## 2. Data Understanding

### 2.1 Data Collection

- **Source:**  
  - A curated 4-class image dataset used for educational and experimental purposes (e.g., a subset of a public dataset or a custom dataset).
- **Size:**  
  - Several hundred images divided across **4** classes.
- **Format:**  
  - RGB images, resized to **224×224** pixels for training and inference, stored as standard image files (e.g., PNG/JPEG).

### 2.2 Data Description

Each sample consists of:

- An input RGB image.
- A categorical label indicating one of the four classes (e.g., `class_0`, `class_1`, `class_2`, `class_3`).

Class distribution is approximately balanced, with each class having a similar number of examples. This makes accuracy and macro F1-score meaningful summary metrics.

### 2.3 Data Quality

During initial exploration we checked for:

- Missing or corrupted files (non-readable images).
- Obvious annotation errors (images that clearly belong to a different class).
- Class imbalance issues.

Any corrupted images were removed. Labels that looked suspicious during inspection were checked and corrected. Overall, the dataset quality is sufficient for training a small classification model.

---

## 3. Data Preparation (Excerpt)

### 3.2 Preprocessing

The same preprocessing steps are implemented both in the training notebook and in the deployed backend so that inference is consistent with training.

### 3.3 Data Augmentation

To improve generalization and reduce overfitting we apply the following augmentations **only on the training set**:

- **Random horizontal flip** – makes the model invariant to left–right orientation.  
- **Random resized crop around 224×224** – simulates small translations and zoom.  
- (Optionally) **mild color jitter or small rotations** to mimic different lighting conditions and viewpoints.

Validation and test images are **not** augmented; they only go through deterministic resizing and normalization. In the notebook we show that augmentation improves validation performance in our ablation studies.

---

## 4. Modeling

### 4.1 Model Choice

We use **ResNet-18** as our primary architecture:

- Pretrained on ImageNet and then fine-tuned on our 4-class dataset.
- The final fully-connected layer is replaced with a new linear layer producing 4 outputs (one per class).

We also experiment with alternative configurations, such as:

- Training the same architecture **from scratch** (no pretraining).  
- Trying slightly different backbones (e.g., a deeper ResNet variant) as part of our ablation studies.

### 4.2 Training Configuration

Our default training configuration is:

- **Loss function:** Cross-entropy loss  
  - Suitable for multi-class classification with mutually exclusive labels.
- **Optimizer:** Adam  
  - Adaptive learning rate; works well with minimal tuning.
- **Learning rate:** `1e-3`  
- **Batch size:** `64`  
- **Epochs:** around **20** epochs, with early stopping or selection of the best checkpoint based on validation accuracy.

### 4.3 Hyperparameter Tuning & Ablation Studies

We explore several variations:

- Learning rate values: `1e-2`, `1e-3`, `1e-4`  
- With vs without **data augmentation**  
- **Pretrained ResNet-18** vs training from **scratch**  
- (Optionally) deeper architectures such as **ResNet-34**

Results of these experiments are summarized in the *Experiments and Results* section of the main report and in tables inside the training notebook. In general, augmentation and pretraining both improve performance, while poorly chosen learning rates can lead to unstable or slow training.

---

## 5. Evaluation

### 5.1 Metrics

We evaluate on the held-out test set using:

- **Accuracy** – overall fraction of correctly classified images.  
- **Precision, Recall, F1-score per class** – useful for understanding performance on each of the four classes.  
- **Macro and weighted F1-score** – to summarize performance across all classes.  
- **Confusion matrix** – to inspect which classes are commonly confused.

These metrics are visualized as:

- Training vs validation **loss curves**  
- Training vs validation **accuracy curves**  
- **Confusion matrix** heatmap  
- **Classification report** (text)  
- (Optionally) **ROC / PR curves** for binary or one-vs-rest setups

### 5.2 Results Summary

For our best configuration (pretrained ResNet-18 with augmentation, learning rate `1e-3`, batch size 64), we obtain:

- **Test accuracy:** ~84%  
- **Macro and weighted F1-score:** ~0.84  

Key observations:

- Data augmentation improves validation and test accuracy compared to training without augmentation.  
- Using pretrained weights from ImageNet yields better results than training from scratch on the small 4-class dataset.  
- Learning rates that are too high or too low both degrade performance relative to the chosen default of `1e-3`.

### 5.3 Error Analysis

We inspect typical failure cases by:

- Visualizing **misclassified test images** in a grid, with true and predicted labels.  
- Examining rows/columns of the **confusion matrix** for systematic confusions.

Common issues include:

- Confusion between visually similar classes (e.g., `class_1` vs `class_2`).  
- Misclassifications when objects are small, heavily occluded, or appear in cluttered backgrounds.

These analyses help guide potential future improvements, such as collecting more data for specific classes or exploring models with better localization capabilities.

---

## 6. Deployment

### 6.1 Model Export

After training, we export the model weights to:

```text
exported_model/visionai_model.pth
```

## 6.2 Inference Service

We expose the model via a backend service (e.g., FastAPI or Flask). The service:

- Receives an image from the VisionAI frontend (file upload or base64).
- Applies preprocessing identical to the training pipeline (resize, tensor conversion, normalization).
- Runs inference with the trained model to obtain class probabilities.
- Returns a JSON response containing:
  - Predicted class labels  
  - Confidence scores (probabilities)  
  - Any additional structured outputs (e.g., top-k predictions or extra metadata)

The VisionAI web client calls this API using a configurable environment variable:

```bash
VITE_API_BASE_URL
```

## 6.3 End-to-End System

The full VisionAI pipeline is:

1. Training & Evaluation

Run the training notebook to retrain and evaluate the model, and to regenerate metrics plots in artifacts/metrics/:

```bash
notebooks/visionai_model_training.ipynb
```
