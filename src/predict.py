import tensorflow as tf
import numpy as np
from pathlib import Path


# ============================================
# SETTINGS
# ============================================

MODEL_PATH = "models/fire_smoke_cnn.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["Fire", "Neutral", "Smoke"]


# ============================================
# LOAD MODEL
# ============================================

print("\n================================")
print("LOADING CNN MODEL")
print("================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index] * 100

    print("\n================================")
    print("PREDICTION RESULT")
    print("================================")

    print(f"Image      : {image_path}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nClass probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions
    ):
        print(
            f"{class_name:<10}: "
            f"{probability * 100:.2f}%"
        )


# ============================================
# TEST IMAGE
# ============================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter image path: "
    ).strip()

    if not Path(image_path).exists():

        print("\nERROR: Image not found!")

    else:

        predict_image(image_path)