from fastapi import APIRouter, HTTPException
import google.generativeai as genai
from schemas.model_schemas import VideoInput, EvaluationResponse
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 📌 팀장 스타일에 맞게 APIRouter 사용
router = APIRouter()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_3d_model(data: VideoInput):
    """
    동영상 URL을 받아서 Gemini 2.0으로 3D 모델 품질을 평가하는 API
    """
    prompt = f"""
    3D 모델이 회전하는 동영상입니다. 다음 기준으로 평가하세요:
    - 텍스처 품질
    - 형태 정밀함
    - 대칭성
    - 전반적인 완성도

    동영상 URL: {data.video_url}
    """
    try:
        response = model.generate_content(prompt)
        return {"evaluation": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API 오류: {str(e)}")