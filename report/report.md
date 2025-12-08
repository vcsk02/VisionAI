# VisionAI: An End-to-End Image Classification System with Web-Based Inference

**Authors:**  
- Vineeth Chandra Sai Kandukuri  
- Siddharth Rao Kartik  
- Alekya Gudise  

---

## Abstract

Deep learning models for computer vision have achieved impressive performance, but they are often difficult to train, evaluate, and deploy in user-facing applications. In this project we build **VisionAI**, an end-to-end image classification system that combines a convolutional neural network with a web-based interface for interactive inference.

We follow the CRISP-DM methodology: starting from problem definition and data understanding, through data preparation and modeling, to evaluation and deployment. Our model is based on a ResNet-18 architecture pretrained on ImageNet and fine-tuned on a 4-class image dataset. Images are resized to 224×224, normalized using standard ImageNet statistics, and augmented during training with random horizontal flips and random resized crops.

The final model achieves approximately 84% accuracy on a held-out test set, with macro and weighted F1-scores of about 0.84. We analyze training and validation curves, confusion matrices, and classification reports, and we run ablation studies to quantify the impact of data augmentation, pretraining, and learning rate choices. We also visualize failure cases to understand common error patterns.

The trained model is exported as a PyTorch checkpoint and served by a lightweight backend (e.g., FastAPI or Flask), which exposes a `/predict` API endpoint. The VisionAI frontend, implemented as a Vite-based web app, allows users to upload images and view predicted labels and confidence scores. Overall, VisionAI demonstrates how a relatively simple transfer learning model can be turned into a reproducible, evaluable, and deployable end-to-end system.

---

## 1. Introduction

Computer vision systems are increasingly used in real-world applications such as medical diagnosis, autonomous driving, content moderation, and industrial inspection. Despite this progress, building a complete pipeline—from data and model training to evaluation and deployment—remains non-trivial, especially for students and practitioners who are new to the field.

In this project we aim to:

1. Train a robust image classification model on a small, labeled, 4-class image dataset.  
2. Evaluate the model with clear metrics and visualizations to understand its behavior.  
3. Deploy the model behind a web-based interface (VisionAI) that supports interactive inference on user-supplied images.

We cast the task as multi-class image classification and adopt a ResNet-18 backbone pretrained on ImageNet. We follow the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to structure the project, emphasizing reproducibility and documentation.

Our main contributions are:

- An end-to-end training pipeline implemented in a Jupyter/Colab notebook (`notebooks/visionai_model_training.ipynb`) that can retrain the model from scratch.  
- A set of evaluation artifacts (loss/accuracy curves, confusion matrix, classification report, and failure examples) stored in `artifacts/metrics/` and `artifacts/failures/`.  
- A simple deployment stack where the trained model is served by a backend API and consumed by a Vite-based web frontend.  
- An empirical evaluation showing that our best model achieves roughly 84% accuracy and balanced F1-scores across four classes on a held-out test set.

The remainder of this report is organized as follows. Section 2 reviews related work. Section 3 describes the dataset and preprocessing. Section 4 presents our modeling approach. Section 5 details experiments and results, including ablations and failure analysis. Section 6 concludes and suggests future extensions.

---

## 2. Related Work

### 2.1 Deep Learning for Image Classification

Convolutional neural networks (CNNs) have become the standard approach for image classification tasks. Architectures such as AlexNet, VGG, and ResNet demonstrated that deep networks can learn hierarchical visual features directly from data. In particular, ResNet introduced residual connections, enabling the training of much deeper networks without severe degradation of gradients.

Pretrained models on large-scale datasets such as ImageNet are commonly used as feature extractors, with the final layers adapted to specific downstream tasks. This transfer learning strategy is especially effective when the target dataset is relatively small, which is the case in our project.

### 2.2 Tools for Training and Deployment

Frameworks like PyTorch and TensorFlow provide flexible APIs for defining and training deep models, as well as utilities for data loading, augmentation, and model checkpointing. Higher-level libraries and ecosystem tools simplify the implementation of training loops and evaluation procedures.

On the deployment side, lightweight web frameworks (e.g., Flask, FastAPI) are often used to wrap trained models as REST APIs, and modern frontend frameworks (React, Vite, etc.) enable user-friendly interfaces over these APIs. Tools like Gradio and Streamlit also support rapid ML demos, though they are less integrated into fully custom production-style stacks.

### 2.3 Positioning of VisionAI

VisionAI is not intended to push the state of the art in classification accuracy. Instead, it serves as a compact example of:

- Using a pretrained ResNet model for transfer learning on a small custom dataset.  
- Following a structured methodology (CRISP-DM) from problem definition to deployment.  
- Instrumenting the model with meaningful metrics and visualizations.  
- Deploying the resulting model behind a real web interface for interactive inference.

Our work is similar in spirit to existing model demo and deployment tools, but it focuses on building a **custom, transparent, and educational** end-to-end pipeline that students can inspect, understand, and modify.

---

## 3. Data

### 3.1 Dataset Description

We work with a small 4-class image dataset (referred to as the **VisionAI image dataset**). Each sample consists of:

- An RGB image.  
- A categorical label from one of four classes: `class_0`, `class_1`, `class_2`, and `class_3`.

The images are stored as standard image files (e.g., `.jpg`, `.png`). In total, we work with several hundred labeled images; for the current experiments, the test split consists of 200 images (50 per class).

### 3.2 Data Splits

We split the dataset into three disjoint subsets:

- **Training set:** used to learn model parameters.  
- **Validation set:** used for hyperparameter tuning and model selection.  
- **Test set:** used only once at the end for final evaluation.

The splits are stratified by class where possible to preserve class proportions. The test set contains 200 images (50 per class). The remaining images are divided into training and validation sets, with the training set significantly larger than the validation set to give the model enough data to learn.

### 3.3 Preprocessing

All images are preprocessed using the same steps:

1. Resize to 224×224 pixels.  
2. Convert to a PyTorch tensor with shape `(3, 224, 224)`.  
3. Normalize each channel using ImageNet mean and standard deviation:  

   ```python
   mean = (0.485, 0.456, 0.406)
   std  = (0.229, 0.224, 0.225)
   ```
These preprocessing steps are implemented both in the training notebook and in the backend service to keep training and inference consistent.

### 3.4 Data Augmentation

To improve generalization and reduce overfitting, we apply random augmentations only to training images:

Random horizontal flip.
Random resized crop centered around 224×224.
(Optionally) mild color jitter or small rotations.

Validation and test images use no random augmentation and only undergo resizing and normalization. Ablation experiments (Section 5.4) show that training without augmentation leads to increased overfitting and lower validation/test performance.

## 4. Methods
### 4.1 Problem Formulation

We treat the task as multi-class image classification. Given an input image

x∈R3×224×224
x∈R
3×224×224
,
the model outputs a probability distribution over four classes. The predicted class is the argmax of this distribution.

### 4.2 Model Architecture

Our primary model is a ResNet-18 network:

Initialized with ImageNet-pretrained weights.

The final fully connected layer is replaced by a new linear layer with 4 outputs, one per class.

ReLU activations and Batch Normalization are used as in the original ResNet design.

We chose ResNet-18 because it offers a good balance between capacity and computational efficiency, and because pretrained weights are readily available. For a relatively small dataset, transfer learning with a moderately sized backbone is a practical and effective approach.

### 4.3 Loss Function and Optimization

We use cross-entropy loss, which is standard for multi-class classification with mutually exclusive labels. The loss compares the model’s predicted probability distribution to the one-hot encoded ground truth.

For optimization, we use the Adam optimizer with:

Learning rate: 
1×10−3
1×10
−3

Default Adam parameters for β-values and epsilon

Adam provides per-parameter adaptive learning rates and generally converges faster than vanilla SGD on this type of problem.

### 4.4 Hyperparameters

Key hyperparameters include:

Learning rates explored: 
1×10−2
1×10
−2
, 
1×10−3
1×10
−3
, 
1×10−4
1×10
−4

Batch size: 64

Number of epochs: around 20

We performed small-scale experiments on the validation set to determine that 
1×10−3
1×10
−3
 is a good default learning rate. Higher learning rates (e.g., 
1×10−2
1×10
−2
) led to unstable training, while lower ones (e.g., 
1×10−4
1×10
−4
) converged more slowly and achieved lower validation accuracy within the same epoch budget.

###    4.5 CRISP-DM Perspective

From a CRISP-DM viewpoint:

Business understanding: Provide an accessible, end-to-end vision model and demo that illustrates the full lifecycle of a computer vision system.

Data understanding & preparation: Analyze, split, preprocess, and augment a 4-class dataset.

Modeling: Fine-tune a pretrained ResNet-18 with appropriate hyperparameters.

Evaluation: Use metrics and visualizations to assess performance and limitations.

Deployment: Wrap the model behind a backend API and integrate it into a web frontend.

This structure encourages clear documentation and makes the pipeline easier to reproduce and extend.

## 5. Experiments and Results
### 5.1 Training Behavior

We track training and validation loss and accuracy per epoch. The corresponding plots (loss_curve.png, accuracy_curve.png) are stored in artifacts/metrics/.

The curves show that:

Training loss decreases steadily, and training accuracy increases over epochs.

Validation loss also decreases and then mostly stabilizes.

The gap between training and validation accuracy is modest, indicating some overfitting but generally acceptable generalization.

### 5.2 Final Test Performance

On the held-out test set of 200 images (50 per class), our best model achieves:

Accuracy: ~84%

Macro F1-score: ~0.84

Weighted F1-score: ~0.84

Approximate per-class performance is as follows:

class_0: precision 0.88, recall 0.90, F1-score 0.89

class_1: precision 0.78, recall 0.80, F1-score 0.79

class_2: precision 0.79, recall 0.76, F1-score 0.78

class_3: precision 0.90, recall 0.90, F1-score 0.90

These results are consistent with the classification report stored in artifacts/metrics/classification_report.txt and suggest relatively balanced performance across classes.

### 5.3 Confusion Matrix and Failure Analysis

We compute and visualize a confusion matrix for the test set (confusion_matrix.png in artifacts/metrics/). The diagonal entries are large, indicating many correct predictions, but we observe noticeable confusion between certain class pairs, particularly class_1 and class_2.

To better understand model failures, we visualize a small set of misclassified examples in artifacts/failures/misclassified_examples.png. Common failure modes include:

Visually similar classes under challenging lighting or viewpoints.

Small objects occupying only a small portion of the image.

Cluttered or distracting backgrounds.

These qualitative analyses complement our quantitative metrics and highlight directions for potential improvement, such as collecting more diverse training data or exploring architectures with better localization capabilities.

### 5.4 Ablation Studies

We perform several ablation experiments to quantify the impact of key design choices. A representative summary is:

Without augmentation vs. with augmentation:

Without augmentation: lower validation and test accuracy, stronger overfitting (training accuracy much higher than validation accuracy).

With augmentation: improved validation and test metrics, smoother learning curves, and reduced overfitting.

Pretrained vs. from scratch:

Pretrained ResNet-18: best test accuracy (~84%) and F1-score (~0.84).

Training from scratch: noticeably lower performance and slower convergence within 20 epochs.

Learning rate variations:

1×10−2
1×10
−2
: unstable training, validation loss oscillates.

1×10−3
1×10
−3
: best performance and convergence speed.

1×10−4
1×10
−4
: stable but underfits within 20 epochs.

These ablations confirm that data augmentation, pretraining, and a reasonable learning rate are all important for good performance on this dataset.

## 6. Conclusion and Future Work

In this project we implemented VisionAI, an end-to-end image classification system that spans:

Data preparation and augmentation.

Training and evaluation of a ResNet-18 model.

Visualization of metrics and failure cases.

Deployment as a backend API consumed by a web frontend.

Following the CRISP-DM methodology helped structure our work from problem definition to deployment. Our best model achieves roughly 84% test accuracy and balanced F1-scores across four classes, demonstrating that a relatively simple transfer learning setup can perform well on a small dataset when combined with appropriate data augmentation and tuning.

The deployment stack shows how to connect a trained PyTorch model to a real user-facing interface, turning a notebook experiment into an interactive application.

There are several natural directions for future work:

Data: Collect more training data and increase diversity in viewpoints, lighting conditions, and backgrounds.

Models: Experiment with alternative architectures (e.g., deeper ResNet variants, ConvNeXt, or Vision Transformers) and compare performance and efficiency.

Localization and robustness: Explore models or techniques that better handle small objects and cluttered scenes, such as attention mechanisms or object detection–style approaches.

Deployment: Add monitoring, logging, and basic model health checks to the backend, and extend the frontend to support batch predictions and richer visual explanations (e.g., Grad-CAM visualizations).
