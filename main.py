from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
import os
import gdown

# Model 1
url1 = "https://drive.google.com/uc?id=1bTqxlRXHfPpcA1U2FK9rV72Q7eQuURCU"
output1 = "Trained_Model.keras"

if not os.path.exists(output1):
    print("Downloading model 1...")
    gdown.download(url1, output1, quiet=False)

model1 = tf.keras.models.load_model(output1)
print("Model 1 loaded")


# Model 2
url2 = "https://drive.google.com/uc?id=1DGnFfpa-rsmJ65d0elUmCZiU3G1DiFvW"
output2 = "Trained_Model_2.keras"

if not os.path.exists(output2):
    print("Downloading model 2...")
    gdown.download(url2, output2, quiet=False)

model2 = tf.keras.models.load_model(output2)
print("Model 2 loaded")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load class indices1
with open("class_names.json") as f:
    class_indices1 = json.load(f)

# Reverse mapping1
index_to_class1 = {v: k for k, v in class_indices1.items()}

# Load class indices2
with open("class_names_2.json") as f:
    class_indices2 = json.load(f)

# Reverse mapping2
index_to_class2 = {v: k for k, v in class_indices2.items()}


# Load remedies1
with open("remedies_38_class.json", "r") as f:
    remedies1 = json.load(f)

# Load remedies2
with open("remedies_47_class.json","r") as f:
    remedies2 = json.load(f)


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

        # Model 1 prediction
        pred1 = model1.predict(processed_image)
        index1 = int(np.argmax(pred1))
        confidence1 = float(np.max(pred1))
        class1 = index_to_class1[index1]

        # Model 2 prediction
        pred2 = model2.predict(processed_image)
        index2 = int(np.argmax(pred2))
        confidence2 = float(np.max(pred2))
        class2 = index_to_class2[index2]

        # Choose prediction with higher confidence
        if confidence1 > confidence2:
            predicted_class = class1
            confidence = confidence1
        else:
            predicted_class = class2
            confidence = confidence2

        # Get remedy from correct dataset
        if predicted_class in remedies1:
            remedies = remedies1[predicted_class]
        elif predicted_class in remedies2:
            remedies = remedies2[predicted_class]
        else:
            remedies = {
                "name": predicted_class,
                "description": "No detailed information available.",
                "remedies": [],
                "precautions": [],
                "prevention": []
            }

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "remedies": remedies
        }

    except Exception as e:
        return {"error": str(e)}


