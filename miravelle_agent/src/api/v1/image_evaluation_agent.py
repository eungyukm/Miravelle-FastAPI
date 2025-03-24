import io
import requests
from fastapi import APIRouter, HTTPException
from src.services.get_evaluation_image import get_evaluation_image_random
import clip
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
from base64 import b64encode
from langchain.agents import tool

router = APIRouter()

# Select device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define NIMA model
class NIMA(nn.Module):
    def __init__(self, base_model):
        super(NIMA, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

# Load NIMA model
nima_base = resnet50(weights=ResNet50_Weights.DEFAULT)
nima_model = NIMA(nima_base).to(device)
nima_model.eval()

def get_clip_score(image, input_text="default"):
    text_features = clip_model.encode_text(clip.tokenize(input_text).to(device))
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image_tensor)

    # Calculate cosine similarity
    similarity = torch.cosine_similarity(text_features, image_features).item()
    clip_score = similarity * 5 + 5

    # Limit score between 1 and 10
    return max(1, min(10, clip_score))

def get_nima_score(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = nima_model(image_tensor).cpu().numpy()[0]
    mean_score = np.dot(scores, np.arange(1, 11))
    return round(mean_score, 2)

@router.get("/v1/evaluate-random-image")
def evaluate_random_image():
    try:
        image_data = get_evaluation_image_random()
        if not image_data:
            raise HTTPException(status_code=404, detail="Failed to load image")

        image = Image.open(io.BytesIO(image_data))
        if image.mode == "RGBA":
            image = image.convert("RGB")

        clip_score = get_clip_score(image)
        nima_score = get_nima_score(image)
        encoded_image = b64encode(image_data).decode("utf-8")

        return {
            "clip_score": clip_score,
            "nima_score": nima_score,
            "image_data": f"data:image/jpeg;base64,{encoded_image}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate image: {str(e)}")

# LangChain Tool
@tool
def get_image_from_api(tool_input=None):
    """
    Fetch a random image from the API, then return scores and Base64-encoded image
    """
    url = "http://127.0.0.1:8000/v1/evaluate-random-image"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        clip_score = data.get("clip_score")
        nima_score = data.get("nima_score")
        encoded_image = data.get("image_data")
        return f"CLIP Score: {clip_score}, NIMA Score: {nima_score}\nImage: {encoded_image}"
    else:
        return "Failed to load image"

@router.get("/v1/get-image/")
def get_image():
    result = get_image_from_api.invoke({})
    if "Image" in result:
        clip_score_line = result.split("\n")[0]
        encoded_image_line = result.split("\n")[1]

        clip_score = clip_score_line.split(",")[0].replace("CLIP Score: ", "")
        nima_score = clip_score_line.split(",")[1].replace(" NIMA Score: ", "")
        encoded_image = encoded_image_line.replace("Image: ", "")

        return {
            "clip_score": clip_score,
            "nima_score": nima_score,
            "image_data": encoded_image
        }
    else:
        return {"error": "Failed to load image"}
