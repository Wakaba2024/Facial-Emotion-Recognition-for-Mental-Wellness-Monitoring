# 🎭 Facial Emotion Recognition API & Web App

## Overview

This project is a **Facial Emotion Recognition (FER) system** designed to detect human emotions from facial images and provide insights for **mental wellness applications**.  
It combines **deep learning (TensorFlow/Keras)** for model training, a **Flask API** for serving predictions, and is deployed on **Render** for cloud accessibility.  

You can interact with it via:  
- 🌐 **Web Form** → Upload an image & see prediction

- ## 🚀 Features
- - Detects **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
- Train a **CNN model** on the FER dataset (from https://datasets.activeloop.ai/docs/ml/datasets/fer2013-dataset/).  
- Save & load the model (`.keras` format).  
- Flask API with endpoints:  
  - `/` → Health check  
  - `/predict` → JSON prediction endpoint  
  - `/upload` → Web UI for uploading images  
- **Deployed on Render** for public access  
- Supports both **pip** and **uv** for dependency management.  

---

## 📂 Project Structure
Facial_Emotion/

│── app.py # Flask app

│── my_model.keras # Trained CNN model

│── static/uploads/ # Uploaded images stored here

│── requirements.txt # Dependencies

│── Procfile # (for Render deployment)

│── README.md # The Project documentation

│── FER_Mental_Wellness.ipynb # Training & model building

---

## 📦 Dependencies  

This project uses [uv](https://github.com/Wakaba2024/UV-Python) for dependency management.  
All dependencies are defined in **`pyproject.toml`**:  

```toml
[project]
name = "facial-emotion-api"
version = "0.1.0"
description = "Facial Emotion Recognition API with Flask + TensorFlow"
authors = [{ name = "Your Name" }]
dependencies = [
    "flask",
    "tensorflow",
    "opencv-python",
    "numpy>=1.23,<2.3",
    "gunicorn"
]
```

---

## ⚙️ Usage and Setup  

1. Install [uv](https://docs.astral.sh/uv/getting-started/):  
  
2. Clone the project:  
   ```bash
   git clone https://github.com/your-username/facial-emotion-api.git
   cd facial-emotion-api
   ```

3. Run locally:  
   ```bash
   uv run python app.py
   ```
   

---

---

## ☁️ Deployment  

### Render  (https://render.com/docs)

- Push to GitHub → Connect to Render → Deploy.  


---


## Results  

From the training and evaluation in the notebook file:  

- **Training Accuracy:** ~92%  
- **Validation Accuracy:** ~66%  
- **Test Accuracy:** ~65%  
- **Loss Trend:** Training loss decreased steadily, validation loss plateaued after ~20 epochs.  

### Confusion Matrix Insights  
- Best performance on **Happy** and **Angry**  
- Most misclassifications between **Sad ↔ Neutral** and **Fear ↔ Surprise**  

### Example Predictions  
- Smiling face → **Happy (95% confidence)**  
- Frowning face → **Sad (88% confidence)**  
- Neutral face → **Neutral (81% confidence)**  

---

## Deployment on Render  

This project is live on **Render**, making the model accessible via the cloud.  

### Deployment Steps  

1. Push your project to GitHub  
2. Create a new **Web Service** on [Render](https://render.com/)  
3. Connect your GitHub repository  
4. In the Render dashboard:  
5. Render provides a public URL (e.g., `https://fer-mental-wellness.onrender.com`)  

---






