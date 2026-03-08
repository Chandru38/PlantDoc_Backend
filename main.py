from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
import os
import gdown
import tensorflow as tf

url = "https://drive.google.com/uc?id=1bTqxlRXHfPpcA1U2FK9rV72Q7eQuURCU"
output = "Trained_Model.keras"

# Download model if it does not exist
if not os.path.exists(output):
    print("Downloading model...")
    gdown.download(url, output, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(output)
print("Model loaded successfully")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
# model = tf.keras.models.load_model("Trained_Model.keras")
print("Model Output Shape:", model.output_shape)

# Load class indices
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
index_to_class = {v: k for k, v in class_indices.items()}

# Load remedies
with open("remedies_38_class.json", "r") as f:
    remedies_data = json.load(f)


def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        predicted_class = index_to_class[predicted_index]

        remedies = remedies_data.get(predicted_class, {
            "name": predicted_class,
            "description": "No detailed information available.",
            "remedies": [],
            "precautions": [],
            "prevention": []
        })

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "remedies": remedies
        }

    except Exception as e:

        return {"error": str(e)}
