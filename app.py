from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the saved model (must be in the same folder as app.py)
model = tf.keras.models.load_model("my_model.keras")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_image(image_bytes):
    """Preprocess image before feeding into the model"""
    # Convert bytes to numpy array
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)  # grayscale
    img = cv2.resize(img, (48, 48))                  # FER input size
    img = img.astype("float32") / 255.0              # normalize
    img = np.expand_dims(img, axis=-1)               # add channel dim
    img = np.expand_dims(img, axis=0)                # add batch dim
    return img

# ✅ Home route
@app.route("/", methods=["GET"])
def home():
    return "✅ Facial Emotion Recognition API is running! Use /predict (POST) or /upload (form)."

# ✅ Upload form route
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        img = preprocess_image(file.read())

        preds = model.predict(img)
        label = emotion_labels[np.argmax(preds)]

        return f"Predicted Emotion: <b>{label}</b>"

    # Render simple HTML form
    return render_template_string("""
        <!doctype html>
        <title>Upload Image for Emotion Prediction</title>
        <h1>Upload an image</h1>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Predict">
        </form>
    """)

# ✅ API route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = preprocess_image(file.read())

    preds = model.predict(img)
    label = emotion_labels[np.argmax(preds)]

    return jsonify({"emotion": label})

if __name__ == "__main__":
    print("✅ Flask app is starting...")
    app.run(debug=True, port=5000)
