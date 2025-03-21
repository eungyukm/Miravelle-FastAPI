from fastapi import APIRouter, Response
from services.get_evaluation_image import get_evaluation_image_random

from langchain.tools import tool
import requests
from base64 import b64encode

router = APIRouter()

@router.get("/v1/get-random-image")
def get_random_image():
    image_data = get_evaluation_image_random()

    if image_data:
        # 이미지 반환
        return Response(content=image_data, media_type="image/jpeg")
    else:
        return {"error": "Failed to load image"}

@tool
def get_image_from_api(tool_input=None):
    """
    REST API에서 랜덤 이미지를 가져옵니다.
    """
    url = "http://127.0.0.1:8000/v1/get-random-image"
    response = requests.get(url)
    if response.status_code == 200:
        encoded_image = b64encode(response.content).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_image}"
    else:
        return "Failed to load image"

@router.get("/v1/get-image/")
def get_image():
    # LangChain Tool 실행 시 invoke() 사용 및 tool_input 전달 필요
    result = get_image_from_api.invoke({})
    if result.startswith("data:image/jpeg;base64"):
        # Base64 인코딩된 이미지 반환
        base64_data = result.split(",")[1]
        image_data = b64encode(base64_data.encode('utf-8')).decode('utf-8')
        return {"image_data": f"data:image/jpeg;base64,{image_data}"}
    else:
        return {"error": "Failed to load image"}