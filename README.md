# 🔥 Fire & Smoke Detection using CNN

A deep learning-based image classification project that uses a Convolutional Neural Network (CNN) to classify images into three categories:

* 🔥 Fire
* 🌫️ Smoke
* 🌳 Neutral

The project demonstrates the complete machine learning workflow, including dataset validation, preprocessing, CNN model development, training, evaluation, visualization, and single-image prediction.

---

## 📌 Project Overview

Early detection of fire and smoke can help reduce damage and improve safety.

This project investigates the use of a custom Convolutional Neural Network to automatically classify images as:

**Fire / Smoke / Neutral**

The model is trained using a labeled image dataset and evaluated using multiple classification metrics.

---

## 🎯 Objectives

The main objectives of this project are:

1. Prepare and validate an image classification dataset.
2. Verify that images can be decoded correctly by TensorFlow.
3. Analyze the distribution of images across classes.
4. Build a custom CNN architecture.
5. Apply image augmentation to improve generalization.
6. Train the CNN using training and validation data.
7. Evaluate the trained model on unseen test data.
8. Analyze performance using:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix
9. Perform prediction on individual images.

---

## 🗂️ Dataset

The dataset contains three classes:

| Class     | Training Images | Test Images |
| --------- | --------------: | ----------: |
| Fire      |             900 |         100 |
| Neutral   |             900 |         100 |
| Smoke     |             900 |         100 |
| **Total** |        **2700** |     **300** |

Total images:

**3,000**

### Dataset Split

The 2,700 training images are divided using an 80/20 split:

* Training: 2,160 images
* Validation: 540 images
* Testing: 300 images

The test set is kept separate from the training/validation data.

> The raw dataset is not included in this GitHub repository.

---

## 🔍 Dataset Validation

Two validation checks were performed before training.

### Image Dataset Check

All 3,000 images were checked.

```text
Images checked: 3000
Bad images found: 0

All images are valid!
```

### TensorFlow Image Check

The complete dataset was also tested using TensorFlow image decoding.

```text
Images checked: 3000
Bad images found: 0

All images can be decoded by TensorFlow!
```

This confirmed that the images were suitable for the TensorFlow training pipeline.

---

## 🧠 CNN Architecture

A custom CNN architecture was developed for the classification task.

### Architecture

```text
Input Image
   ↓
Data Augmentation
   ├── Random Flip
   ├── Random Rotation
   └── Random Zoom
   ↓
Convolution Block 1
   ├── Conv2D: 32 filters
   ├── Batch Normalization
   ├── ReLU
   └── Max Pooling
   ↓
Convolution Block 2
   ├── Conv2D: 64 filters
   ├── Batch Normalization
   ├── ReLU
   └── Max Pooling
   ↓
Convolution Block 3
   ├── Conv2D: 128 filters
   ├── Batch Normalization
   ├── ReLU
   └── Max Pooling
   ↓
Convolution Block 4
   ├── Conv2D: 256 filters
   ├── Batch Normalization
   ├── ReLU
   └── Max Pooling
   ↓
Global Average Pooling
   ↓
Dense Layer: 128
   ↓
Dropout
   ↓
Softmax
   ↓
Fire / Neutral / Smoke
```

### Model Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Input Size        | 224 × 224 × 3                   |
| Number of Classes | 3                               |
| Optimizer         | Adam                            |
| Learning Rate     | 0.001                           |
| Loss Function     | Sparse Categorical Crossentropy |
| Activation        | ReLU + Softmax                  |
| Regularization    | Batch Normalization + Dropout   |
| Pooling           | Global Average Pooling          |
| Data Augmentation | Flip + Rotation + Zoom          |

The improved CNN contains approximately:

**423,139 parameters**

---

## 🏋️ Training

The CNN was trained for:

**30 epochs**

using:

* Batch size: 32
* Image size: 224 × 224
* Training samples: 2,160
* Validation samples: 540

The best validation performance was obtained at:

```text
Best Epoch: 10
Best Validation Accuracy: 87.41%
```

Training and validation accuracy/loss were visualized using training curves.

---

## 📊 Test Evaluation

The final model was evaluated using the separate 300-image test dataset.

### Classification Report

```text
              precision    recall  f1-score   support

Fire           0.7719    0.8800    0.8224       100
Neutral        0.6517    0.5800    0.6138       100
Smoke          0.6598    0.6400    0.6497       100

accuracy                           0.7000       300
macro avg      0.6945    0.7000    0.6953       300
weighted avg   0.6945    0.7000    0.6953       300
```

### Test Accuracy

**70.00%**

---

## 🔢 Confusion Matrix

The final confusion matrix was:

```text
[[88  5  7]
 [16 58 26]
 [10 26 64]]
```

Interpretation:

| Actual Class | Correct Predictions |
| ------------ | ------------------: |
| Fire         |            88 / 100 |
| Neutral      |            58 / 100 |
| Smoke        |            64 / 100 |

The model performed best on the **Fire** class, while Neutral and Smoke showed more confusion with each other.

---

## 🔮 Single Image Prediction

The project includes a prediction script that loads the trained CNN and predicts the class of an individual image.

Example:

```bash
python src/predict.py
```

The program accepts an image path and displays:

```text
PREDICTION RESULT

Image      : <image path>
Prediction : Smoke
Confidence : 100.00%

Class probabilities:
Fire      : 0.00%
Neutral   : 0.00%
Smoke     : 100.00%
```

The prediction confidence represents the model's softmax output and should not be interpreted as a guarantee of correctness.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* Pillow
* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
fire_smoke_detection/
│
├── dataset/                    # Local dataset - not uploaded
│
├── models/                    # Trained model - not uploaded
│
├── src/
│   ├── check_images.py
│   ├── check_tensorflow_images.py
│   ├── dataset_summary.py
│   ├── prepare_dataset.py
│   ├── build_cnn.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_training.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd fire_smoke_detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 1. Check Images

```bash
python src/check_images.py
```

### 2. Check TensorFlow Image Decoding

```bash
python src/check_tensorflow_images.py
```

### 3. View Dataset Summary

```bash
python src/dataset_summary.py
```

### 4. Prepare Dataset

```bash
python src/prepare_dataset.py
```

### 5. Build CNN

```bash
python src/build_cnn.py
```

### 6. Train Model

```bash
python src/train.py
```

### 7. Evaluate Model

```bash
python src/evaluate.py
```

### 8. Plot Training Performance

```bash
python src/plot_training.py
```

### 9. Predict an Individual Image

```bash
python src/predict.py
```

---

## ⚠️ Important Note About the Dataset

The dataset is intentionally excluded from this repository through `.gitignore`.

To reproduce the project, download the appropriate Fire/Smoke/Neutral image dataset and place it in:

```text
dataset/FIRE-SMOKE-DATASET/
```

with the following structure:

```text
FIRE-SMOKE-DATASET/
├── Train/
│   ├── Fire/
│   ├── Neutral/
│   └── Smoke/
│
└── Test/
    ├── Fire/
    ├── Neutral/
    └── Smoke/
```

---

## ⚠️ Limitations

Although the model achieved strong validation performance, the final test accuracy was approximately 70%.

Important limitations include:

* Limited dataset size.
* Possible visual similarity between smoke and neutral scenes.
* Variation in lighting and environmental conditions.
* Model may not generalize to all real-world fire scenarios.
* High prediction confidence does not necessarily mean the prediction is correct.
* The model is intended as a demonstration/research project and should not be treated as a certified fire safety system.

---

## 🔮 Future Improvements

Possible improvements include:

* Transfer learning using MobileNetV2, EfficientNet, or ResNet.
* Increasing dataset diversity.
* Hyperparameter tuning.
* Class-specific augmentation.
* Learning-rate scheduling.
* Early stopping and model checkpointing.
* Cross-validation.
* Grad-CAM visualization for model interpretability.
* Real-time webcam/video detection.
* Deployment as a web application.
* Deployment as a mobile application.
* Comparison between custom CNN and transfer-learning models.

---

## 📌 Conclusion

This project demonstrates an end-to-end deep learning workflow for Fire, Smoke, and Neutral image classification.

The workflow includes:

**Dataset Validation → Dataset Preparation → CNN Architecture → Training → Evaluation → Visualization → Prediction**

The custom CNN achieved:

* **87.41% best validation accuracy**
* **70.00% final test accuracy**

The project also demonstrates the importance of evaluating a deep learning model using class-wise precision, recall, F1-score, and confusion matrices rather than relying only on training accuracy.

---

## 👨‍💻 Author

**Vimal**

B.Tech Artificial Intelligence & Data Science

---

## ⭐ Project Status

**Completed — CNN-based Fire & Smoke Image Classification**
