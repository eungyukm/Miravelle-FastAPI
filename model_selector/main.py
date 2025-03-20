import os
from fastapi import FastAPI
import redis
import clip
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np

app = FastAPI()
nima_base = resnet50(weights=ResNet50_Weights.DEFAULT)

# Redis 연결
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# NIMA 모델 정의 (미적 품질 평가)
class NIMA(nn.Module):
    def __init__(self, base_model):
        super(NIMA, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, 10)  # 10개의 점수 예측

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

# NIMA 모델 로드
nima_base = resnet50(pretrained=True)
nima_model = NIMA(nima_base).to(device)
nima_model.eval()

# 3D 모델 폴더
image_folder = "generated_3d_models"

# 유사도 평가 함수
def get_clip_score(image_path, input_text):
    text_features = clip_model.encode_text(clip.tokenize(input_text).to(device))
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image)
    similarity = torch.cosine_similarity(text_features, image_features)
    return similarity.item() * 100  # 100점 기준 변환

# 품질 평가 함수 (NIMA)
def get_nima_score(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path)
    
    # RGBA (4채널) → RGB (3채널) 변환
    if image.mode == "RGBA":
        image = image.convert("RGB")

    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        scores = nima_model(image).cpu().numpy()[0]
    
    mean_score = np.dot(scores, np.arange(1, 11))  # 평균 점수 계산
    return mean_score * 10  # 100점 기준 변환

# 최적 모델 선택
def get_best_model(image_files, input_text):
    best_model = None
    best_final_score = -1

    for image_path in image_files:
        # 이미지 파일만 처리 ('.DS_Store' 같은 파일 제외)
        if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"무시된 파일: {image_path}")  # 디버깅 로그
            continue

        clip_score = get_clip_score(image_path, input_text)
        nima_score = get_nima_score(image_path)

        # 필터 조건
        if nima_score < 70:
            continue  

        final_score = (nima_score * 0.8) + (clip_score * 0.2)

        if final_score > best_final_score:
            best_final_score = final_score
            best_model = image_path

    return best_model, best_final_score

# API 엔드포인트 (Redis 적용)
@app.get("/select-best-model/")
async def select_best_model(keyword: str):
    print(f"🔥 요청받은 키워드: {keyword}")  # 키워드 확인

    # 이미지 파일 목록 불러오기
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    image_files = [img for img in image_files if img.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"📂 찾은 이미지 파일 목록: {image_files}")  # 이미지 파일이 제대로 로드되는지 확인

    best_model, best_score = get_best_model(image_files, keyword)

    if best_model:
        print(f"최적 모델 선택: {best_model} (점수: {best_score})")  # 최종 선택된 모델
        return {"best_model": best_model, "score": best_score}
    else:
        print("적절한 모델을 찾지 못함")  # 모든 모델이 필터링되었는지 확인
        return {"message": "No suitable model found"}


# 서버 실행: uvicorn main:app --reload
""""
http://127.0.0.1:8000/select-best-model/?keyword=cartoon-style+elf+mage
여기로 들어가면 generated_3d_models 폴더에 있는 이미지 중에 제일 괜찮은 모델이 나와용"
"""