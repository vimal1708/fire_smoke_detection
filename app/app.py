import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request,send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/fire_smoke_cnn.keras"
UPLOAD_FOLDER = "app/uploads"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["Fire", "Neutral", "Smoke"]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# ==========================================
# CREATE UPLOAD FOLDER
# ==========================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# LOAD MODEL
# ==========================================

print("\n================================")
print("LOADING FIRE & SMOKE CNN")
print("================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# CHECK FILE
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    probabilities = None
    image_path = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:

            error = "Please select an image."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                error=error
            )

        if not allowed_file(file.filename):

            error = "Only JPG, JPEG and PNG images are allowed."

            return render_template(
                "index.html",
                error=error
            )

        # --------------------------------------
        # SAVE IMAGE
        # --------------------------------------

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        # --------------------------------------
        # LOAD IMAGE
        # --------------------------------------

        image = Image.open(filepath).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # --------------------------------------
        # PREDICTION
        # --------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = np.argmax(predictions)

        prediction = CLASS_NAMES[predicted_index]

        confidence = float(
            predictions[predicted_index] * 100
        )

        # --------------------------------------
        # CLASS PROBABILITIES
        # --------------------------------------

        probabilities = []

        for class_name, probability in zip(
            CLASS_NAMES,
            predictions
        ):

            probabilities.append({
                "name": class_name,
                "value": float(probability * 100)
            })

        # Browser image path
        image_path = "/uploads/" + filename
        
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        image_path=image_path,
        error=error
    )

# ==========================================
# SERVE UPLOADED IMAGES
# ==========================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("FIRE & SMOKE DETECTION UI")
    print("================================")

    print("Open: http://127.0.0.1:5000")

    app.run(debug=True)