# VisionAI – Model Inputs, Outputs, and Metrics

This document defines the input/output interface of the VisionAI model and the metrics we use to evaluate it. It is meant to be the "contract" between the dataset, the training code, the backend API, and the frontend UI.

---

## 1. Inputs

### 1.1 Data Type

- **Modality:** Images
- **Format:** RGB images (3 channels)
- **File types:** [e.g., `.jpg`, `.png`]

### 1.2 Shape and Resolution

Before feeding images into the model, we:

- Resize to **[e.g., 224 × 224]** pixels.
- Represent each image as a tensor of shape:

```text
(batch_size, 3, H, W)
