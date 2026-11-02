
import cv2
import tensorflow as tf
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# Settings
MODEL_PATH = "models/improved_fire_smoke_cnn.keras"
DATASET_PATH = Path("dataset/FIRE-SMOKE-DATASET/Train")
IMAGE_SIZE = (224, 224)
CSV_PATH = Path("detection_history.csv")

# Get class names in training order
CLASS_NAMES = sorted(
    folder.name for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
)

# Load model
print("Loading Fire & Smoke Detection Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded!")
print("Classes:", CLASS_NAMES)

# Create CSV if it doesn't exist
if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Prediction",
            "Confidence"
        ])

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access webcam.")
    exit()

print("Webcam started. Press Q to quit.")

last_prediction = None

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # Preprocess image
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb_frame,
        IMAGE_SIZE
    )

    image_array = np.expand_dims(
        resized.astype("float32"),
        axis=0
    )

    # Predict
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    # Save only when predicted class changes
    if predicted_class != last_prediction:
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(CSV_PATH, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                predicted_class,
                round(confidence, 2)
            ])

        print(
            f"{timestamp} | {predicted_class} | "
            f"{confidence:.2f}%"
        )

        last_prediction = predicted_class

    # Display prediction
    label = f"{predicted_class}: {confidence:.1f}%"

    if predicted_class.lower() == "fire":
        color = (0, 0, 255)
    elif predicted_class.lower() == "smoke":
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow(
        "Fire & Smoke Detection - Press Q to Exit",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nDetection history saved to: {CSV_PATH}")
