import requests
from langchain.tools import tool
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool(return_direct=True)
def get_image_from_api():
    """
    랜덤 이미지를 가져와서 평가합니다.
    """
    try:
        url = "http://127.0.0.1:8000/v1/evaluate-random-image"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            clip_score = data.get("clip_score")
            nima_score = data.get("nima_score")
            encoded_image = data.get("image_data")

            logger.info(f"CLIP 점수: {clip_score}, NIMA 점수: {nima_score}")
            return f"CLIP 점수: {clip_score}, NIMA 점수: {nima_score}\n이미지: {encoded_image}"
        else:
            logger.error(f"이미지 다운로드 실패: 상태 코드 {response.status_code}")
            return "이미지를 불러오지 못했습니다."
    except Exception as e:
        logger.error(f"랜덤 이미지 다운로드 실패: {str(e)}")
        return f"이미지 다운로드 실패: {str(e)}"
