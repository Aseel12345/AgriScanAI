import os
import gdown
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Download model if not exists
MODEL_PATH = "model/crop_weed_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1IlSnN3kTpZZ5Lqjs9---Vopwfyqfhfk_"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded ✅")

# Load the model with compatibility flags
model = None
try:
    print("Loading model...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False  # Required for some h5 models
    )
    print("Model loaded successfully ✅")
except Exception as e:
    print(f"Error loading model: {e}")

app = Flask(__name__)
CORS(app)

# STEP 1: Use This Mapping
CLASSES = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent",
    "Maize", "Scentless Mayweed", "Shepherds Purse",
    "Small-flowered Cranesbill", "Sugar beet"
]

# STEP 2: Add Crop vs Weed Logic
def map_to_crop_or_weed(class_name):
    crops = ["Common wheat", "Maize", "Sugar beet"]
    if class_name in crops:
        return "Crop"
    else:
        return "Weed"

def prepare_image(image_file):
    """Preprocess the image for the model."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224)) # Adjust size to match your model's input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        # Process the image
        processed_image = prepare_image(file)

        # Make prediction if model is loaded
        if model:
            predictions = model.predict(processed_image)
            class_index = predictions.argmax()
            class_name = CLASSES[class_index]
            final_label = map_to_crop_or_weed(class_name)

            # Return Response
            return jsonify({
                "label": final_label,
                "plant_type": class_name,
                "confidence": float(predictions[0][class_index])
            })
        else:
            return jsonify({"error": "Model not loaded on server"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
