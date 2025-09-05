# 🎭 Facial Emotion Recognition API & Web App

This project is a **deep learning-based facial emotion recognition system** built with **TensorFlow/Keras**, **Flask**, and **OpenCV**.  
It allows users to upload an image and receive the predicted **emotion** (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).  

You can interact with it via:  
- 🌐 **Web Form** → Upload an image & see prediction

- ## 🚀 Features
- Train a **CNN model** on the FER dataset (from https://datasets.activeloop.ai/docs/ml/datasets/fer2013-dataset/).  
- Save & load the model (`.keras` format).  
- Flask API with endpoints:  
  - `/` → Health check  
  - `/predict` → JSON prediction endpoint  
  - `/upload` → Web UI for uploading images  
- Deployable on **Render**, **Hugging Face Spaces**, or **Docker**.  (Render Was Used For This Application)
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

## ⚙️ Setup  

1. Install [uv](https://docs.astral.sh/uv/getting-started/):  
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```  
   or on Windows:  
   ```powershell
   powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone the project:  
   ```bash
   git clone https://github.com/your-username/facial-emotion-api.git
   cd facial-emotion-api
   ```

3. Sync dependencies:  
   ```bash
   uv sync
   ```

4. Run locally:  
   ```bash
   uv run python app.py
   ```
   App will be available at:  
   👉 http://127.0.0.1:5000  

---

---

## ☁️ Deployment  

### Render  (https://render.com/docs)

- Push to GitHub → Connect to Render → Deploy.  


---


## 📊 Monitoring & Logs  
- Flask prints logs to console (visible in Render dashboard).  




