import os
from fastapi import APIRouter
import clip
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np

router = APIRouter()

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# NIMA 모델 정의 (미적 품질 평가)
class NIMA(nn.Module):
    def __init__(self, base_model):
        super(NIMA, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

# NIMA 모델 로드
nima_base = resnet50(weights=ResNet50_Weights.DEFAULT)
nima_model = NIMA(nima_base).to(device)
nima_model.eval()

# 3D 모델 폴더
image_folder = "generated_3d_models"

# CLIP 유사도 평가 함수
def get_clip_score(image_path, input_text):
    text_features = clip_model.encode_text(clip.tokenize(input_text).to(device))

    image = Image.open(image_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    image = preprocess(image).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image)
    similarity = torch.cosine_similarity(text_features, image_features)

    return similarity.item() * 100  # 100점 기준 변환

# NIMA 품질 평가 함수
def get_nima_score(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = nima_model(image).cpu().numpy()[0]

    mean_score = np.dot(scores, np.arange(1, 11))
    final_score = mean_score * 10  # 100점 기준 변환

    return final_score

# 최적 모델 선택 함수
def get_best_model(image_files, input_text):
    best_model = None
    best_final_score = -1
    backup_model = None

    for image_path in image_files:
        clip_score = get_clip_score(image_path, input_text)
        nima_score = get_nima_score(image_path)

        # 백업 모델 설정 (가장 높은 점수 기록)
        if backup_model is None or nima_score > get_nima_score(backup_model):
            backup_model = image_path

        # 50점 미만이면 제외
        if nima_score < 50:
            continue  

        final_score = (nima_score * 0.8) + (clip_score * 0.2)

        if final_score > best_final_score:
            best_final_score = final_score
            best_model = image_path

    # 후보 모델이 없을 경우 백업 모델 반환
    if best_model is None:
        if backup_model:
            best_model = backup_model
            best_final_score = get_nima_score(backup_model)
        else:
            return {"best_model": None, "score": None, "message": "No suitable model found"}

    message = "최적 모델이 선택되었습니다."
    if best_final_score < 55:
        message += " 하지만 선택된 모델의 점수가 낮아 품질이 완벽하지 않을 수 있습니다."

    return {
        "best_model": best_model,
        "score": round(best_final_score, 2),
        "message": message
    }

# API 엔드포인트
@router.get("/select-best-model/")
async def select_best_model(keyword: str):
    image_files = [
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg")) and not img.startswith(".")
    ]

    if not image_files:
        return {"message": "No images found in the directory"}

    result = get_best_model(image_files, keyword.strip())

    if result["best_model"]:
        return result
    else:
        return {"message": "No suitable model found"}