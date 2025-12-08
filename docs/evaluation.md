# VisionAI – Model Evaluation

This document explains how we evaluate the VisionAI model, how the dataset is split, what metrics we use, and how to interpret the plots in `artifacts/metrics/`.

---

## 1. Dataset Splits

We partition our dataset into three disjoint sets:

- **Train:** ~[X%] of the data  
- **Validation:** ~[Y%]  
- **Test:** ~[Z%]

The **training set** is used to fit model parameters.  
The **validation set** is used for model selection and hyperparameter tuning.  
The **test set** is held out until the end and used only for final reporting.

Splits are [stratified / random], ensuring that each class is reasonably represented in all splits.

---

## 2. Data Augmentation

To improve generalization and reduce overfitting, we apply the following augmentations on the training set only:

- **Random horizontal flip:** Helps the model handle mirrored images.
- **Random crop / resized crop:** Adds robustness to small translations and zoom.
- **[Any others you use: color jitter, rotation, etc.]**

The validation and test sets receive **no augmentation**, only deterministic preprocessing (resize, normalization), so metrics truly reflect generalization.

---

## 3. Core Metrics

We use the following metrics to evaluate the model:

- **Accuracy:**  
  Fraction of correctly classified examples. Easy to interpret and a good overall indicator when classes are relatively balanced.

- **Precision, Recall, F1-score (per class):**  
  Useful when classes are imbalanced or when some error types are more costly than others.  
  - Precision: of all predicted positives, how many are correct?  
  - Recall: of all true positives, how many did we find?  
  - F1: harmonic mean of precision and recall.

- **Confusion matrix:**  
  Shows which classes are being confused with others and highlights systematic errors.

For some settings we also consider **[mAP / IoU / ROC-AUC]** if relevant to the task.

---

## 4. Plots and What They Show

All figures are stored in `artifacts/metrics/`.

### 4.1 Training vs Validation Loss (`loss_curve.png`)

- **Goal:** Check for overfitting and convergence.
- **Healthy behavior:**
  - Training and validation loss both decrease and then stabilize.
- **Warning signs:**
  - Training loss keeps dropping while validation loss increases → overfitting.
  - Loss fluctuates heavily → learning rate might be too high or batch size too small.

### 4.2 Training vs Validation Accuracy (`accuracy_curve.png`)

- **Goal:** Track how predictive performance evolves.
- **Healthy behavior:**
  - Both curves increase and plateau.
  - Validation accuracy is close to training accuracy.
- **Warning signs:**
  - Big gap between train and val accuracy → overfitting.
  - Both curves low → model underfits or architecture is too weak.

### 4.3 Confusion Matrix (`confusion_matrix.png`)

- **Goal:** Understand which classes are confused.
- The diagonal cells represent correct predictions; off-diagonal cells are misclassifications.
- Patterns we observe:
  - **[Example: “Class A and B are often confused, likely because they look visually similar.”]**
  - **[Example: “Class C has very low recall, probably due to few training examples.”]**

### 4.4 Classification Report (`classification_report.txt`)

- Contains precision, recall, F1-score, and support for each class.
- Used to:
  - Identify minority classes with poor performance.
  - Compare different model variants during ablation studies.

### 4.5 Optional ROC / PR Curves (`roc_curve.png`, etc.)

- **ROC curve:** Shows trade-off between true positive rate and false positive rate.
- **PR curve:** More informative than ROC under heavy class imbalance.
- Area under the curve (AUC) summarizes overall ranking quality.

---

## 5. Interpretation & Key Findings

From the metrics and plots we conclude:

- The best model configuration is **[model + hyperparams]**, achieving:
  - Test accuracy: **[X%]**
  - Macro F1-score: **[Y]**  
- Data augmentation **[e.g., improved validation accuracy by Z% and reduced overfitting]**.
- Main failure modes:
  - **[Examples: misclassification under heavy occlusion / tiny objects / similar-looking classes.]**

These insights guide future improvements, such as collecting more data for difficult classes, trying more powerful architectures, or tuning augmentation strategies.
