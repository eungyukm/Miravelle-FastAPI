from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini 설정
if not GOOGLE_API_KEY:
    raise ValueError("Google API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI 앱 생성
app = FastAPI()

# 입력 데이터 모델
class VideoInput(BaseModel):
    video_url: str

# 평가 텍스트에서 점수 추출 함수
def parse_scores(response_text):
    pattern = r"(텍스처|형태|대칭성|완성도)[^\d]*(\d+(?:\.\d+)?)/10"
    matches = re.findall(pattern, response_text)
    scores = {label: float(score) for label, score in matches}

    # 요약 코멘트 추출 (첫 번째 줄 또는 마지막 줄 기준)
    lines = response_text.strip().splitlines()
    summary = lines[-1] if lines else "요약 없음"

    return {
        "scores": scores,
        "summary": summary,
        "raw": response_text
    }

# 평가 함수
def analyze_video(url):
    prompt = f"""
    다음은 3D 모델을 회전시켜 보여주는 동영상입니다.
    이 모델의 품질을 다음 항목을 기준으로 평가해주세요:
    1. 텍스처 품질
    2. 형태의 정밀함
    3. 대칭성
    4. 전반적인 완성도
    평가 결과는 각 항목별 점수(10점 만점)와 한 줄 요약 코멘트로 구성해주세요.

    동영상 URL: {url}
    """
    try:
        response = model.generate_content(prompt)
        return parse_scores(response.text)
    except Exception as e:
        return {"error": f"Gemini API 호출 중 오류 발생: {str(e)}"}

# 평가 요청 엔드포인트
@app.post("/evaluate/")
async def evaluate_3d_model(data: VideoInput):
    result = analyze_video(data.video_url)
    return result
