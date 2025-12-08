# VisionAI – Evaluation

This document explains how we evaluate the VisionAI model and how to interpret the figures in `artifacts/metrics/`.

---

## 1. Dataset Splits

We split the dataset into three parts:

- **Train:** used to fit model parameters  
- **Validation:** used for hyperparameter tuning and model selection  
- **Test:** used only once, for final reporting

In our experiments we use approximately:

- Train: 75%  
- Validation: 8%  
- Test: 17%  

The validation set is never used to update model weights; it is only used to pick hyperparameters and the best checkpoint. The test set remains untouched until the end to provide an unbiased estimate of performance.

---

## 2. Data Augmentation

To improve generalization and reduce overfitting, we apply data augmentation **only to the training set**. Typical augmentations include:

- `RandomHorizontalFlip` – makes the model robust to left–right flips  
- `RandomResizedCrop` / `RandomCrop` – simulates small translations and zoom  
- (Optional) `ColorJitter`, mild `RandomRotation`, etc.

The **validation** and **test** sets use only deterministic preprocessing:

1. Resize to the target resolution (e.g. 224×224)  
2. Convert to tensor  
3. Normalize by channel mean and standard deviation  

This ensures the reported metrics reflect true generalization, not randomness from augmentation.

---

## 3. Core Metrics

We evaluate the model on the held-out test set using the following metrics.

### 3.1 Accuracy

Overall accuracy is:

\[
\text{Accuracy} = \frac{\text{# correct predictions}}{\text{total # samples}}
\]

This is an intuitive, single-number summary of performance, especially when classes are reasonably balanced.

### 3.2 Precision, Recall, and F1-score

For each class \( c \):

- **True Positives (TP\_c)** – predicted as \( c \) and actually \( c \)  
- **False Positives (FP\_c)** – predicted as \( c \) but actually not \( c \)  
- **False Negatives (FN\_c)** – actually \( c \) but predicted as something else  

We compute:

\[
\text{Precision}(c) = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}
\]

\[
\text{Recall}(c) = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
\]

\[
\text{F1}(c) = 2 \cdot \frac{\text{Precision}(c) \cdot \text{Recall}(c)}{\text{Precision}(c) + \text{Recall}(c)}
\]

We use `sklearn.metrics.classification_report` to compute per-class precision/recall/F1 plus macro and weighted averages.

### 3.3 Confusion Matrix

We also compute a confusion matrix \( C \) where:

- Rows = true classes  
- Columns = predicted classes  
- \( C[i, j] \) = number of samples of class \( i \) predicted as class \( j \)

This reveals which classes are systematically confused and complements the scalar metrics above.

### 3.4 Optional Metrics

For some tasks we may also consider:

- **ROC curve / AUC** – for binary or one-vs-rest setups  
- **Task-specific metrics** such as mAP or IoU (for detection/segmentation)

These are only used when relevant to the dataset and architecture.

---

## 4. Plots and Files in `artifacts/metrics/`

All visualizations and reports are stored under `artifacts/metrics/`.

### 4.1 `loss_curve.png` – Training vs Validation Loss

Shows how loss evolves over epochs:

- A **healthy** model typically has both training and validation loss decreasing and then roughly flattening.  
- If training loss keeps dropping while validation loss rises, the model is overfitting.

### 4.2 `accuracy_curve.png` – Training vs Validation Accuracy

Shows accuracy over epochs:

- Ideally, training and validation accuracy both increase and then plateau.  
- A large gap between training and validation accuracy suggests overfitting.  
- Both curves staying low suggests underfitting or an overly simple model.

### 4.3 `confusion_matrix.png` – Confusion Matrix

Visualizes test-time confusions:

- The diagonal cells represent correct predictions.  
- Off-diagonal cells show which classes are mislabeled as which.

This is the main tool for spotting systematically hard classes.

### 4.4 `classification_report.txt` – Precision/Recall/F1

Contains the full text classification report from scikit-learn:

- Per-class precision, recall, F1-score, and support  
- Macro / weighted averages  
- Overall accuracy

We use this to compare variants in ablation studies and to describe results in the written report.

### 4.5 (Optional) `roc_curve.png` – ROC Curve

If we evaluate a binary (or one-vs-rest) setup, this file shows:

- True Positive Rate vs False Positive Rate at different thresholds  
- Area under the curve (AUC) as a scalar summary

---

## 5. Example Results (Current Best Model)

For our current best model on the test set (200 images, 4 classes with 50 samples each), we obtain the following classification report:

- **Accuracy:** 0.84 (84%)  
- **Macro F1-score:** 0.84  
- **Weighted F1-score:** 0.84  

Per-class performance:

- `class_0`: precision 0.88, recall 0.90, F1-score 0.89 (50 samples)  
- `class_1`: precision 0.78, recall 0.80, F1-score 0.79 (50 samples)  
- `class_2`: precision 0.79, recall 0.76, F1-score 0.78 (50 samples)  
- `class_3`: precision 0.90, recall 0.90, F1-score 0.90 (50 samples)  

Interpretation:

- The model is **reasonably balanced** across classes, with all F1-scores in the 0.78–0.90 range.  
- `class_0` and `class_3` are the easiest classes (highest F1).  
- `class_1` and `class_2` are slightly harder; they are more often confused with each other, which is visible in the confusion matrix.

These numbers come from `artifacts/metrics/classification_report.txt` and `confusion_matrix.png`.

---

## 6. Ablation Studies and Failure Analysis (Summary)

We also perform ablation experiments and qualitative error analysis (described in detail in the training notebook and main report):

- **Ablations**
  - With vs without data augmentation  
  - Pretrained backbone vs training from scratch  
  - Different learning rates  
  Augmentation and pretraining consistently improve validation and test accuracy, while overly small or large learning rates hurt performance.

- **Failure cases**
  - We visualize a subset of misclassified test images in `artifacts/failures/misclassified_examples.png`.  
  - Common failure modes include confusion between visually similar classes and images with cluttered backgrounds or atypical viewpoints.

Together, the curves, confusion matrix, classification report, ablations, and failure cases provide a complete picture of how well the model works and where it still struggles.
