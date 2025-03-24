from langchain.tools import tool

from fastapi import HTTPException
import httpx
import logging

# service
from services.image_evaluation import get_image_from_miravell
from dotenv import load_dotenv

# tools
from agent.huggingface_tool import upload_to_huggingface
from agent.get_image_tool import get_image_from_api
from agent.evaluate_miravelle_tool import evaluate_and_save_miravell

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://miravelle-appservice-dsecega7bbhvefem.koreacentral-01.azurewebsites.net"

@tool(return_direct=True, args_schema=None)
def get_image_from_miravell_tool():
    """
    miravelle에서 이미지를 가져옵니다.
    - 예시: "miravelle에서 이미지 가져와"
    """
    try:
        result = get_image_from_miravell()
        return f"API 응답: {result}"
    except Exception as e:
        return f"실패: {str(e)}"