# Autonomous Clinical Triage for Diabetic Retinopathy

## Overview
Diabetic retinopathy silently damages retinal vasculature in diabetic patients and often progresses to irreversible vision loss before obvious clinical symptoms emerge. Screening programs worldwide are under pressure as the number of patients requiring periodic retinal examination keeps rising, while specialist graders have not increased at the same pace.

This project presents an **autonomous clinical triage system** that accepts color fundus images and returns a referral decision through a secure web interface. It is designed as a **first-level screening aid** for resource-limited settings, reducing grading workload and improving triage efficiency.

---

## Key Features
- **Lightweight CNN Architecture**  
  - Compact convolutional neural network with three convolutional blocks  
  - Trained from scratch (no large pretrained backbone)  
  - Preprocessing pipeline includes:
    - Spatial resizing  
    - LAB-space CLAHE enhancement  
    - Per-pixel normalization  

- **Deployment via Django REST API**  
  - Ensures consistent preprocessing between training and inference  
  - Secure web interface for clinical use  
  - Designed for modest infrastructure without GPU dependence  

- **Performance Metrics (Validation Split)**  
  - Accuracy: **0.89**  
  - Precision: **0.87**  
  - Recall: **0.85**  
  - Specificity: **0.91**  
  - F1-score: **0.86**  
  - Training and validation curves showed smooth convergence without instability  

---

## System Workflow
1. **Input**: Color fundus image uploaded via web interface  
2. **Preprocessing**: Resizing, CLAHE enhancement, normalization  
3. **Model Inference**: CNN predicts referral decision (binary classification)  
4. **Output**: Referral recommendation returned securely through Django REST API  

---

## Clinical Positioning
- Not a replacement for ophthalmic diagnosis  
- Functions as a **triage tool** to prioritize patients for specialist review  
- Enables scalable screening in resource-constrained environments  

---

## Installation & Usage
### Requirements
- Python 3.8+  
- Django REST Framework  
- TensorFlow / PyTorch (depending on implementation)  
- OpenCV for image preprocessing  

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/diabetic-retinopathy-triage.git
cd diabetic-retinopathy-triage

# Install dependencies
pip install -r requirements.txt

# Run Django server
python manage.py runserver
