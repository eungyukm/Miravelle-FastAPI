from fastapi import APIRouter, HTTPException
import google.generativeai as genai
from src.schemas.model_schemas import VideoInput, EvaluationResponse
import os
from dotenv import load_dotenv
import re

# .env 파일 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
router = APIRouter()

def parse_response(text: str):
    """
    Gemini 응답 텍스트에서 점수와 요약을 파싱하고 총점 계산
    """
    def extract_score(criterion):
        match = re.search(rf"{criterion}.*?(\d+)/10", text, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    texture_quality = extract_score("텍스처 품질")
    shape_accuracy = extract_score("형태 정밀함")
    symmetry = extract_score("대칭성")
    overall_quality = extract_score("전반적인 완성도")

    total_score = round((texture_quality + shape_accuracy + symmetry + overall_quality) / 4, 2)

    # 요약 문장 추출 (임시로 마지막 문단 전체 사용)
    summary_match = re.search(r"요약.*?:\s*(.+)", text)
    summary = summary_match.group(1).strip() if summary_match else "요약을 찾을 수 없습니다."

    return {
        "texture_quality": texture_quality,
        "shape_accuracy": shape_accuracy,
        "symmetry": symmetry,
        "overall_quality": overall_quality,
        "total_score": total_score,
        "summary": summary,
        "raw_text": text  # 참고용 원문
    }

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_3d_model(data: VideoInput):
    """
    동영상 URL을 받아서 Gemini로 3D 모델 품질을 평가하는 API
    """
    prompt = f"""
    3D 모델이 회전하는 동영상입니다. 다음 기준으로 평가하고 각 항목마다 /10 점수와 간단한 이유를 제공하세요:
    - 텍스처 품질
    - 형태 정밀함
    - 대칭성
    - 전반적인 완성도
    마지막으로 요약 의견도 제공하세요.

    동영상 URL: {data.video_url}
    """
    try:
        response = model.generate_content(prompt)
        parsed = parse_response(response.text)
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API 오류: {str(e)}")