"""
Diabetic Retinopathy Triage System - Flask Application
"""

import os
import pickle
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from model.train_model import extract_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'dr_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_inference(img_path):
    """Run DR inference on an uploaded image."""
    features = extract_features(img_path)
    features_2d = features.reshape(1, -1)

    proba = model.predict_proba(features_2d)[0]
    prediction = int(model.predict(features_2d)[0])
    confidence = float(proba[prediction]) * 100

    # Generate per-channel analysis for UI display
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    r_mean = float(np.mean(enhanced[:, :, 2]))
    g_mean = float(np.mean(enhanced[:, :, 1]))
    b_mean = float(np.mean(enhanced[:, :, 0]))
    edge_density = float(np.sum(cv2.Canny(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), 50, 150) > 0) / (256 * 256))

    return {
        "prediction": prediction,
        "label": "Diabetic Retinopathy Detected" if prediction == 1 else "No Diabetic Retinopathy",
        "confidence": round(confidence, 1),
        "dr_probability": round(float(proba[1]) * 100, 1),
        "normal_probability": round(float(proba[0]) * 100, 1),
        "analysis": {
            "red_channel": round(r_mean, 1),
            "green_channel": round(g_mean, 1),
            "blue_channel": round(b_mean, 1),
            "edge_density": round(edge_density * 100, 2),
        }
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, BMP, or TIFF.'}), 400

    # Save with unique name
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(save_path)

    try:
        result = run_inference(save_path)
        result['image_url'] = f"/static/uploads/{unique_filename}"
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
