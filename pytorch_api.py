from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

transform = T.Compose([T.ToTensor()])

@app.post("/detect/pytorch")
async def detect_pytorch(file: UploadFile = File(...)) -> Dict:
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).to(device)


    with torch.no_grad():
        predictions = model([img_tensor])[0]

    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    print(scores)
    # COCO human label = 1
    human_found = any(label == 1 and score > 0.6 for label, score in zip(labels, scores))

    return {"human_detected": human_found}

@app.get("/test/")
def read_root():
    return {"message": "Hello, World!"}