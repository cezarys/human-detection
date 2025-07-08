from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

def load_image_into_numpy_array(data):
    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    return np.array(image)

@app.post("/detect/tensorflow")
async def detect_tensorflow(file: UploadFile = File(...)) -> Dict:
    img_bytes = await file.read()
    img_np = load_image_into_numpy_array(img_bytes)
    input_tensor = tf.convert_to_tensor([img_np], dtype=tf.uint8)

    detections = detector(input_tensor)
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # COCO person class = 1
    human_found = any(cls == 1 and score > 0.6 for cls, score in zip(classes, scores))

    return {"human_detected": human_found}