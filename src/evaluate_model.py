
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from prepare_dataset import prepare_datasets


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = Path("models/fire_smoke_cnn.keras")


# ==========================================
# START EVALUATION
# ==========================================

print("\n================================")
print("CNN MODEL EVALUATION")
print("================================")


# ==========================================
# 1. CHECK MODEL
# ==========================================

if not MODEL_PATH.exists():
    print("\nERROR: Trained model not found!")
    print(f"Expected location: {MODEL_PATH}")
    raise FileNotFoundError(MODEL_PATH)

print(f"\nLoading model: {MODEL_PATH}")


# ==========================================
# 2. LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 3. PREPARE TEST DATASET
# ==========================================

print("\n================================")
print("LOADING TEST DATA")
print("================================")

_, _, test_dataset, class_names = prepare_datasets()

print("\nClasses:", class_names)


# ==========================================
# 4. EVALUATE TEST DATA
# ==========================================

print("\n================================")
print("TEST PERFORMANCE")
print("================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss     :", round(test_loss, 4))
print("Test Accuracy :", round(test_accuracy, 4))


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

print("\n================================")
print("GENERATING PREDICTIONS")
print("================================")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Predictions generated successfully!")


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ==========================================
# 8. DISPLAY CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title("CNN Confusion Matrix")

plt.tight_layout()

plt.show()


# ==========================================
# 9. PER-CLASS SUMMARY
# ==========================================

print("\n================================")
print("PER-CLASS PERFORMANCE")
print("================================")

for index, class_name in enumerate(class_names):

    actual_count = np.sum(y_true == index)

    correct_count = np.sum(
        (y_true == index) &
        (y_pred == index)
    )

    accuracy = correct_count / actual_count

    print(
        f"{class_name:<10} : "
        f"{correct_count}/{actual_count} "
        f"correct "
        f"({accuracy * 100:.2f}%)"
    )


# ==========================================
# COMPLETE
# ==========================================

print("\n================================")
print("EVALUATION COMPLETE")
print("================================")
