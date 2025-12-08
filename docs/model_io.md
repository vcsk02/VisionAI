# 👁️ VisionAI – Model Inputs, Outputs, and Metrics

This document defines the input/output interface of the **VisionAI 4-class image classification model** (based on **ResNet-18**) and the metrics used to evaluate it.

It acts as the "contract" for consistency between:

* The **dataset** and **training notebook** (`visionai_model_training.ipynb`)
* The **backend inference service** (API)
* The **VisionAI frontend** (UI)

---

## 1. Inputs: Image Specification and Preprocessing

The model is designed to accept standard image data which is then rigorously prepared before inference.

### 1.1 Data Type

* **Modality:** Images
* **Color Space:** **RGB** (3 channels)
* **File Types:** `.jpg`, `.jpeg`, `.png`

### 1.2 Tensor Shape and Resolution

Before entering the model, each image is converted to a PyTorch tensor.

* **Single Image Shape:** $3 \times 224 \times 224$
* **Batch Input Shape:** $(N, 3, 224, 224)$

| Dimension | Value | Description |
| :---: | :---: | :--- |
| **N** | Batch Size | Varies by context (training, inference, etc.) |
| **3** | Channels | RGB color channels |
| **224** | Height | Spatial resolution in pixels |
| **224** | Width | Spatial resolution in pixels |

### 1.3 Preprocessing (Deterministic)

The following pipeline is applied **identically** to **Training, Validation, Test, and Inference** data.

1.  **Resize:** Image is scaled to **$224 \times 224$ pixels**.
2.  **Conversion:** Converted to a PyTorch tensor of shape $(3, 224, 224)$.
3.  **Normalization:** Each channel is normalized using **ImageNet statistics** (zero-mean, unit-variance).

$$
\text{mean} = (0.485, 0.456, 0.406) \\
\text{std} = (0.229, 0.224, 0.225)
$$

### 1.4 Data Augmentation (Training Only)

To improve generalization, random augmentations are applied **only to the training set**:

* **RandomHorizontalFlip:** Encourages left–right flip invariance.
* **RandomResizedCrop** (to $224 \times 224$): Simulates small translations and zoom.
* (Optionally) mild color jitter or small rotations.

Validation, test, and backend inference use only the deterministic preprocessing from 1.3.

---

## 2. Outputs: Model Prediction and API Response

The model performs a **4-class classification** and the results are surfaced via a structured API response.

### 2.1 Raw Model Output

The model outputs a tensor of **logits** (unnormalized scores) for a batch of $N$ images.

* **Logits Shape:** $(N, 4)$
* **Probabilities:** The **softmax** function is applied along the class dimension to obtain a probability distribution $\mathbf{p}$ over the 4 classes, where $p_i$ is the predicted probability for class $i$.

$$
p_i = \frac{e^{\text{logit}_i}}{\sum_{j=0}^{3} e^{\text{logit}_j}}
$$

### 2.2 Class Labels

A fixed, ordered list of labels is maintained consistently across all components (training, backend, frontend):

```python
classes = [
    "class_0",
    "class_1",
    "class_2",
    "class_3"
]
```
You are absolutely right. I missed the final closing remark about the file path and GitHub steps from your original input.

Here is the complete and final README.md content, including the necessary final paragraph:
Markdown

# 👁️ VisionAI – Model Inputs, Outputs, and Metrics

This document defines the input/output interface of the **VisionAI 4-class image classification model** (based on **ResNet-18**) and the metrics used to evaluate it.

It acts as the "contract" for consistency between:

* The **dataset** and **training notebook** (`visionai_model_training.ipynb`)
* The **backend inference service** (API)
* The **VisionAI frontend** (UI)

---

## 1. Inputs: Image Specification and Preprocessing

The model is designed to accept standard image data which is then rigorously prepared before inference.

### 1.1 Data Type

* **Modality:** Images
* **Color Space:** **RGB** (3 channels)
* **File Types:** `.jpg`, `.jpeg`, `.png`

### 1.2 Tensor Shape and Resolution

Before entering the model, each image is converted to a PyTorch tensor.

* **Single Image Shape:** $3 \times 224 \times 224$
* **Batch Input Shape:** $(N, 3, 224, 224)$

| Dimension | Value | Description |
| :---: | :---: | :--- |
| **N** | Batch Size | Varies by context (training, inference, etc.) |
| **3** | Channels | RGB color channels |
| **224** | Height | Spatial resolution in pixels |
| **224** | Width | Spatial resolution in pixels |

### 1.3 Preprocessing (Deterministic)

The following pipeline is applied **identically** to **Training, Validation, Test, and Inference** data.

1.  **Resize:** Image is scaled to **$224 \times 224$ pixels**.
2.  **Conversion:** Converted to a PyTorch tensor of shape $(3, 224, 224)$.
3.  **Normalization:** Each channel is normalized using **ImageNet statistics** (zero-mean, unit-variance).

$$
\text{mean} = (0.485, 0.456, 0.406) \\
\text{std} = (0.229, 0.224, 0.225)
$$

### 1.4 Data Augmentation (Training Only)

To improve generalization, random augmentations are applied **only to the training set**:

* **RandomHorizontalFlip:** Encourages left–right flip invariance.
* **RandomResizedCrop** (to $224 \times 224$): Simulates small translations and zoom.
* (Optionally) mild color jitter or small rotations.

Validation, test, and backend inference use only the deterministic preprocessing from 1.3.

---

## 2. Outputs: Model Prediction and API Response

The model performs a **4-class classification** and the results are surfaced via a structured API response.

### 2.1 Raw Model Output

The model outputs a tensor of **logits** (unnormalized scores) for a batch of $N$ images.

* **Logits Shape:** $(N, 4)$
* **Probabilities:** The **softmax** function is applied along the class dimension to obtain a probability distribution $\mathbf{p}$ over the 4 classes, where $p_i$ is the predicted probability for class $i$.

$$
p_i = \frac{e^{\text{logit}_i}}{\sum_{j=0}^{3} e^{\text{logit}_j}}
$$

### 2.2 Class Labels

A fixed, ordered list of labels is maintained consistently across all components (training, backend, frontend):

```python
classes = [
    "class_0",
    "class_1",
    "class_2",
    "class_3"
]
```
The Top-1 Prediction is determined by finding the index (idx) with the maximum probability:

    Top-1 Index: idx=argmax(p)

    Top-1 Label: classes[idx]

### 2.3 Backend API Response

The backend returns a JSON object for each inference request, typically including the top-k predictions (e.g., top-3) sorted by probability.

```json
{
  "predictions": [
    {
      "label": "class_0",
      "class_index": 0,
      "probability": 0.87
    },
    ...
  ]
}
```

The JSON structure includes: label (human-readable name), class_index (integer index), and probability (softmax confidence score).

## 3. Evaluation Metrics

All metrics are computed on the held-out test set using predicted labels vs. ground-truth labels. The goal is to achieve an overall test accuracy ≈0.84 and corresponding F1-scores.
### 3.1 Accuracy

Accuracy is the simplest metric, representing the fraction of correct classifications.
Accuracy=# correct predications / total # samples

    Reported: Validation accuracy per epoch (for model selection) and the final test accuracy.

### 3.2 Precision, Recall, and F1-score

These metrics provide a deeper understanding of performance for each class (c).
Summary Metrics: We report per-class F1-scores (expected range 0.78–0.90), along with Macro F1 (unweighted mean) and Weighted F1 (support-weighted mean) for overall health.

### 3.3 Confusion Matrix

A 4×4 confusion matrix C is computed and visualized in artifacts/metrics/confusion_matrix.png.

    C[i,j] is the number of samples of true class i that were predicted as class j.

    Purpose: To identify frequently confused class pairs and spot systematic low recall/precision.

### 3.4 Additional Artifacts

The training process also generates and logs key diagnostic plots:

    artifacts/metrics/loss_curve.png: Training vs. validation loss.

    artifacts/metrics/accuracy_curve.png: Training vs. validation accuracy.

    artifacts/metrics/classification_report.txt: Full text report from sklearn.metrics.classification_report.
