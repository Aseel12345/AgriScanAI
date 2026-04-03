import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
# 3. Ensure CORS exists
CORS(app)

# --- Configuration ---
# 1. Updated Model path
MODEL_PATH = 'model/crop_weed_model.h5'
CLASSES = ["Healthy", "Early Blight", "Late Blight", "Leaf Spot", "Weed"]

# Load the model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

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

    # 2. Add error handling (Wrap prediction)
    try:
        # Process the image
        processed_image = prepare_image(file)

        # Make prediction if model exists
        if model:
            predictions = model.predict(processed_image)
            class_idx = np.argmax(predictions[0])
            label = CLASSES[class_idx]
            confidence = float(np.max(predictions[0]))

            # Example logic for recommendations based on label
            recommendations = {
                "Early Blight": {"chemical": "Fungicide X", "dosage": "2g/L"},
                "Late Blight": {"chemical": "Fungicide Y", "dosage": "3g/L"},
                "Weed": {"chemical": "Glyphosate", "dosage": "1.5L/acre"},
                "Healthy": {"chemical": "None", "dosage": "N/A"},
            }

            rec = recommendations.get(label, {"chemical": "Consult Expert", "dosage": "N/A"})

            return jsonify({
                "label": label,
                "chemical": rec["chemical"],
                "dosage": rec["dosage"],
                "confidence": confidence,
                "message": f"Detection successful with {confidence*100:.1f}% confidence."
            })
        else:
            return jsonify({
                "error": f"Model not found at {MODEL_PATH}. Please ensure the model file is uploaded."
            }), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)