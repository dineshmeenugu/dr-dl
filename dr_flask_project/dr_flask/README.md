# Diabetic Retinopathy Triage System 🫀

A Flask web application for binary DR screening from fundus images.

## Project Structure

```
dr_flask/
├── app.py                  # Flask application & API routes
├── requirements.txt        # Python dependencies
├── model/
│   ├── train_model.py      # Feature extractor + demo model builder
│   └── dr_model.pkl        # Trained model (auto-generated)
├── templates/
│   └── index.html          # Frontend UI
└── static/
    └── uploads/            # Saved uploaded images
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the demo model
python model/train_model.py

# 3. Start Flask
python app.py
```

Open **http://localhost:5000** in your browser.

## API

`POST /predict`  
Form-data: `image` (file)

Returns JSON:
```json
{
  "prediction": 1,
  "label": "Diabetic Retinopathy Detected",
  "confidence": 87.3,
  "dr_probability": 87.3,
  "normal_probability": 12.7,
  "analysis": {
    "red_channel": 142.5,
    "green_channel": 98.2,
    "blue_channel": 74.1,
    "edge_density": 4.72
  },
  "image_url": "/static/uploads/<uuid>.jpg"
}
```

## Using a Real Model (Production)

Replace `model/dr_model.pkl` with a model trained on:
- **APTOS 2019 Blindness Detection** (Kaggle)
- **EyePACS** dataset

The `extract_features()` function in `model/train_model.py` can be
replaced with a PyTorch/TensorFlow CNN for higher accuracy.
Expected metrics from paper: Accuracy 0.89, Precision 0.87, Recall 0.85.

## ⚠️ Disclaimer

This is a **first-level screening aid** only — not a substitute for
ophthalmic diagnosis. All results must be verified by a qualified clinician.
