# -*- coding: utf-8 -*-
"""
이 코드는 FastAPI를 사용하여 이미지 평가 및 허깅페이스(Hugging Face) 데이터셋 업로드 기능을 제공합니다.
CLIP 및 NIMA 모델을 활용하여 이미지를 평가하며, LangChain을 통해 에이전트를 초기화하고 명령을 처리합니다.
"""

import os
import io

import httpx
import requests
from fastapi import FastAPI, HTTPException, APIRouter
from dotenv import load_dotenv
from huggingface_hub import HfApi
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, tool, AgentType

import clip
import torch
from PIL import Image
from base64 import b64encode

import json

from schemas.llm_schemas import CommandRequest

# agent tool
from agent.tools import get_image_from_miravell_tool

# service
from services.image_evaluation import get_image_from_miravell
from services.nima_rating import calculate_nima_score

from services.utils import get_evaluation_image_random

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

router = APIRouter()

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
clip_model, preprocess = clip.load("ViT-B/32", device=device)

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
        nima_score = calculate_nima_score(image)
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
    upload_to_huggingface,
    get_image_from_miravell_tool,
]

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Tool 등록 상태 출력
print(f"Registered Tools: {[tool.name for tool in tools]}")

@router.post("/v1/process-command")
def process_command(request: CommandRequest):
    """
    에이전트를 통해 명령을 처리합니다.
    """
    try:
        result = agent.invoke(request.command)
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
        nima_score = calculate_nima_score(image)

        # 허깅페이스에 업로드
        upload_result = upload_to_huggingface.invoke({"score": nima_score, "image_data": image_data})

        return {
            "nima_score": nima_score,
            "upload_result": upload_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 평가 및 업로드에 실패했습니다: {str(e)}")

@router.get("/v1/get-image-from-miravell")
def evaluate_image_from_miravell():
    result = get_image_from_miravell()
    return f"API 응답: {result}"


@router.get("/v1/evaluate-miravelle-image")
def evaluate_miravelle_image():
    """
    Miravelle 이미지를 다운로드하여 CLIP 및 NIMA 점수를 계산합니다.
    """
    try:
        logger.info("➡ Miravelle 이미지 URL 호출 시작")

        # 이미지 URL 및 id 반환
        result = get_image_from_miravell()
        image_url = result.get("image_url")
        item_id = result.get("id")
        logger.info(f"반환된 이미지 ID: {item_id}, URL: {image_url}")

        if not image_url:
            logger.error("이미지 URL이 None으로 반환됨")
            raise HTTPException(status_code=404, detail="이미지를 불러오지 못했습니다.")

        # URL에서 이미지 데이터 다운로드
        logger.info(f"➡ 이미지 다운로드 시작: {image_url}")
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(image_url)
            logger.info(f"HTTP 상태 코드: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"이미지 다운로드 실패: 상태 코드 {response.status_code}")
                raise HTTPException(status_code=response.status_code, detail="이미지 다운로드 실패")

            image_data = response.content
            logger.info(f"이미지 데이터 크기: {len(image_data)} bytes")

            # 이미지 변환 및 평가
            logger.info("➡ 이미지 변환 시작")
            try:
                image = Image.open(io.BytesIO(image_data))
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                logger.info("이미지 변환 성공")
            except Exception as e:
                logger.error(f"이미지 변환 실패: {str(e)}")
                raise HTTPException(status_code=500, detail=f"이미지 변환 실패: {str(e)}")

            # CLIP 및 NIMA 점수 계산
            logger.info("➡ CLIP 및 NIMA 점수 계산 시작")
            try:
                clip_score = get_clip_score(image)
                nima_score = calculate_nima_score(image)
                logger.info(f"CLIP 점수: {clip_score}, NIMA 점수: {nima_score}")
            except Exception as e:
                logger.error(f"점수 계산 실패: {str(e)}")
                raise HTTPException(status_code=500, detail=f"점수 계산 실패: {str(e)}")

            # base64 인코딩
            logger.info("➡ base64 인코딩 시작")
            try:
                encoded_image = b64encode(image_data).decode("utf-8")
                logger.info(f"base64 인코딩 성공: {len(encoded_image)} bytes")
            except Exception as e:
                logger.error(f"base64 인코딩 실패: {str(e)}")
                raise HTTPException(status_code=500, detail=f"base64 인코딩 실패: {str(e)}")

            # 최종 반환 값에 id 포함
            return {
                "id": item_id,
                "clip_score": clip_score,
                "nima_score": nima_score,
                "image_data": f"data:image/jpeg;base64,{encoded_image}"
            }

    except Exception as e:
        logger.error(f"이미지 평가 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 평가에 실패했습니다: {str(e)}")

# 상수 정의
BASE_URL = "https://miravelle-appservice-dsecega7bbhvefem.koreacentral-01.azurewebsites.net"

@router.post("/v1/evaluate-and-save/")
def evaluate_and_save():
    """
    Miravelle 이미지를 평가하고 결과를 저장합니다.
    """
    try:
        # 평가 수행
        logger.info("➡ Miravelle 평가 시작")
        result = evaluate_miravelle_image()
        logger.info(f"평가 결과: {result}")

        # id와 점수를 추출
        item_id = result.get("id")
        clip_score = result.get("clip_score")
        nima_score = result.get("nima_score")

        if not item_id or not clip_score or not nima_score:
            raise HTTPException(status_code=500, detail="평가 결과가 유효하지 않습니다.")

        # 평균값 계산
        evaluation_score = (clip_score + nima_score) / 2
        logger.info(f"평균 점수 계산 완료: 평균 점수 = {evaluation_score:.2f}")

        # 평가 결과 저장 API 호출
        logger.info(f"➡ 평가 결과 저장 시작: ID={item_id}, 점수={evaluation_score:.2f}")
        payload = {"evaluation_score": evaluation_score}

        with httpx.Client(timeout=10.0) as client:
            url = f"{BASE_URL}/api/v1/vision/image/evaluate/{item_id}/"
            response = client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            logger.info(f"상태 코드: {response.status_code}")
            logger.info(f"응답 본문: {response.text}")

            # 상태 코드 200 또는 201을 성공으로 처리
            if response.status_code in (200, 201):
                logger.info(f"평가 결과 저장 성공: 상태 코드={response.status_code}")

                return {
                    "status": "success",
                    "id": item_id,
                    "clip_score": clip_score,
                    "nima_score": nima_score,
                    "evaluation_score": evaluation_score,
                    "message": "Evaluation score uploaded successfully"
                }
            else:
                logger.error(f"저장 실패: 상태 코드 {response.status_code}, 응답: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to save evaluation score: {response.text}"
                )

    except Exception as e:
        logger.error(f"평가 및 저장 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"평가 및 저장에 실패했습니다: {str(e)}")