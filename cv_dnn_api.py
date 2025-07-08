from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model (MobileNet SSD)
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

@app.post("/detect/cv_dnn")
async def detect_cv_dnn(file: UploadFile = File(...)) -> Dict:
    image = await file.read()
    np_img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    human_found = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            human_found = True
            break

    return {"human_detected": human_found}

@app.get("/test/")
def read_root():
    return {"message": "Hello, World!"}