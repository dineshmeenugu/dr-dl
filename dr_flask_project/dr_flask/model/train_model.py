"""
Diabetic Retinopathy Detection - Feature Extraction + Classifier
Uses OpenCV + scikit-learn since PyTorch is not available in this environment.

For PRODUCTION use, replace with a proper CNN (PyTorch/TensorFlow) trained on
the APTOS 2019 or EyePACS dataset.
"""

import numpy as np
import cv2
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def extract_features(img_path_or_array):
    """
    Extract retinal image features relevant to DR detection:
    - Vessel/lesion texture via GLCM-like features
    - Color channel statistics (red channel dominant in DR)
    - Brightness histogram features
    - Edge density (vessel complexity)
    - Green channel variance (hemorrhage proxy)
    """
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path_or_array}")
    else:
        img = img_path_or_array

    img = cv2.resize(img, (256, 256))

    # LAB CLAHE enhancement (matches paper)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    features = []

    # --- Color channel statistics ---
    for ch in range(3):
        channel = enhanced[:, :, ch].astype(np.float32)
        features += [
            float(np.mean(channel)),
            float(np.std(channel)),
            float(np.percentile(channel, 25)),
            float(np.percentile(channel, 75)),
        ]

    # --- Red channel dominance (hemorrhage indicator) ---
    r = enhanced[:, :, 2].astype(np.float32)
    g = enhanced[:, :, 1].astype(np.float32)
    b_ch = enhanced[:, :, 0].astype(np.float32)
    features.append(float(np.mean(r) / (np.mean(g) + 1e-5)))
    features.append(float(np.mean(r) / (np.mean(b_ch) + 1e-5)))

    # --- Histogram features (green channel - clearest vessel contrast) ---
    hist = cv2.calcHist([enhanced], [1], None, [32], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-5)
    features += hist.tolist()

    # --- Edge density (vessel/lesion boundaries) ---
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    features.append(float(np.sum(edges > 0) / edges.size))

    # --- Texture: Local variance in 8x8 blocks ---
    block_vars = []
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            block = gray[i:i+32, j:j+32].astype(np.float32)
            block_vars.append(float(np.var(block)))
    features += block_vars

    # --- Bright lesion detection (exudates: bright spots on green) ---
    _, bright_mask = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    features.append(float(np.sum(bright_mask > 0) / bright_mask.size))

    # --- Dark lesion detection (hemorrhages / microaneurysms) ---
    _, dark_mask = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY_INV)
    features.append(float(np.sum(dark_mask > 0) / dark_mask.size))

    return np.array(features, dtype=np.float32)


def build_demo_model():
    """
    Build and save a DEMO model with synthetic feature patterns.
    
    ⚠️  THIS IS A DEMO MODEL - NOT CLINICALLY VALIDATED.
    Replace with a real model trained on APTOS/EyePACS dataset.
    
    The feature extractor is medically grounded; only the training
    data here is synthetic for demonstration purposes.
    """
    np.random.seed(42)
    n_samples = 400

    # Simulate DR-positive features: higher red dominance, more lesions
    N_FEATURES = 113  # matches extract_features() output
    X_pos = np.random.randn(n_samples // 2, N_FEATURES)
    X_pos[:, 0] += 2.0    # higher mean red
    X_pos[:, 4] += 1.5    # higher std green
    X_pos[:, 8] += 1.0    # red/green ratio up
    X_pos[:, 111] += 0.3  # more bright lesions
    X_pos[:, 112] += 0.2  # more dark lesions

    # Simulate DR-negative features: balanced channels
    X_neg = np.random.randn(n_samples // 2, N_FEATURES)

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        ))
    ])
    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(__file__), 'dr_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Demo model saved to {model_path}")
    return model


if __name__ == "__main__":
    build_demo_model()
