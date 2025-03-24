# -*- coding: utf-8 -*-
"""
이 코드는 FastAPI를 사용하여 이미지 평가 및 허깅페이스(Hugging Face) 데이터셋 업로드 기능을 제공합니다.
CLIP 및 NIMA 모델을 활용하여 이미지를 평가하며, LangChain을 통해 에이전트를 초기화하고 명령을 처리합니다.
"""

import os
import io
import requests
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import HfApi
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, tool, AgentType

import clip
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
from base64 import b64encode

import json

from schemas.llm_schemas import CommandRequest

# 환경 변수 로드
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

router = APIRouter()

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
clip_model, preprocess = clip.load("ViT-B/32", device=device)


# NIMA 모델 정의 및 로드
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


# CLIP 평가 함수
def get_clip_score(image, input_text="default"):
    """
    입력 이미지와 텍스트에 대한 CLIP 점수를 계산합니다.
    """
    text_features = clip_model.encode_text(clip.tokenize(input_text).to(device))
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image_tensor)

    similarity = torch.cosine_similarity(text_features, image_features).item()
    clip_score = similarity * 5 + 5

    # 점수를 1과 10 사이로 제한
    return max(1, min(10, clip_score))


# NIMA 평가 함수
def get_nima_score(image):
    """
    입력 이미지에 대한 NIMA 점수를 계산합니다.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = nima_model(image_tensor).cpu().numpy()[0]
    mean_score = np.dot(scores, np.arange(1, 11))
    return float(round(mean_score, 2))  # float으로 변환하여 반환


# 랜덤 이미지 다운로드 함수
def get_evaluation_image_random():
    response = requests.get("https://picsum.photos/200/300")
    if response.status_code == 200:
        return response.content
    return None


# 평가 엔드포인트
@router.get("/v1/evaluate-random-image")
def evaluate_random_image():
    """
    랜덤 이미지를 다운로드하여 CLIP 및 NIMA 점수를 계산합니다.
    """
    try:
        image_data = get_evaluation_image_random()
        if not image_data:
            raise HTTPException(status_code=404, detail="이미지를 불러오지 못했습니다.")

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
        raise HTTPException(status_code=500, detail=f"이미지 평가에 실패했습니다: {str(e)}")


# 허깅페이스 업로드 함수
@tool(return_direct=True)
def upload_to_huggingface(score, image_data):
    """
    평가 점수와 이미지를 허깅페이스 데이터셋에 업로드합니다.
    """
    api = HfApi()

    # 데이터셋 디렉토리 생성
    os.makedirs("dataset", exist_ok=True)

    # 이미지 파일 저장
    image_file = "dataset/evaluated_image.jpg"
    with open(image_file, "wb") as img_f:
        img_f.write(image_data)

    # 점수 데이터 저장
    data = {"score": score}
    dataset_file = "dataset/nima_scores.jsonl"
    with open(dataset_file, "a") as f:
        f.write(json.dumps(data) + "\n")

    # 허깅페이스에 파일 업로드
    api.upload_file(
        path_or_fileobj=dataset_file,
        path_in_repo="nima_scores.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HUGGINGFACE_TOKEN
    )

    api.upload_file(
        path_or_fileobj=image_file,
        path_in_repo="evaluated_image.jpg",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HUGGINGFACE_TOKEN
    )

    return f"점수 {score}와 이미지가 {DATASET_REPO}에 업로드되었습니다."


# LangChain Tool 설정 (입력값 제거)
@tool(return_direct=True)
def get_image_from_api():
    """
    랜덤 이미지를 가져와서 평가합니다.
    """
    url = "http://127.0.0.1:8000/v1/evaluate-random-image"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        clip_score = data.get("clip_score")
        nima_score = data.get("nima_score")
        encoded_image = data.get("image_data")
        return f"CLIP 점수: {clip_score}, NIMA 점수: {nima_score}\n이미지: {encoded_image}"
    else:
        return "이미지를 불러오지 못했습니다."


# LangChain 설정 (단일 입력값으로 수정)
tools = [
    get_image_from_api,
    upload_to_huggingface
]

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@router.post("/v1/process-command")
def process_command(request: CommandRequest):
    """
    에이전트를 통해 명령을 처리합니다.
    """
    try:
        result = agent.run(request.command)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/evaluate-and-upload")
def evaluate_and_upload():
    """
    랜덤 이미지를 가져와 NIMA로 평가하고, 평가 점수와 이미지를 허깅페이스 데이터셋에 업로드합니다.
    """
    try:
        # 랜덤 이미지 가져오기
        image_data = get_evaluation_image_random()
        if not image_data:
            raise HTTPException(status_code=404, detail="이미지를 불러오지 못했습니다.")

        # 이미지 열기 및 RGB로 변환
        image = Image.open(io.BytesIO(image_data))
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # NIMA 점수 계산
        nima_score = get_nima_score(image)

        # 허깅페이스에 업로드
        upload_result = upload_to_huggingface.invoke({"score": nima_score, "image_data": image_data})

        return {
            "nima_score": nima_score,
            "upload_result": upload_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 평가 및 업로드에 실패했습니다: {str(e)}")

