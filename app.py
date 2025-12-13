import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------------------------
# App Configuration
# ----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "microplastics_model.keras")

model = load_model(MODEL_PATH)

CLASS_NAMES = ["Clean Water", "Microplastics"]

# ----------------------------
# ROUTES
# ----------------------------

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/detector")
def predict_page():
    return render_template("predict.html")


# API for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = CLASS_NAMES[1]
        confidence = prediction
        risk = "High"
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - prediction
        risk = "Low"

    return jsonify({
        "label": label,
        "confidence": round(float(confidence) * 100, 2),
        "risk_level": risk
    })


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
