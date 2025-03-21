from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()
app = FastAPI()


@router.get("/image/test")
def image_test():
    return {"message": "image router is working!"}


class EvaluationResponse(BaseModel):
    texture_quality: dict
    shape_accuracy: dict
    symmetry: dict
    overall_quality: dict
    total_score: float
    summary: str
    improvements: list

@app.post("/evaluate")
async def evaluate_model():
    evaluation_result = {
        "texture_quality": {
            "description": "텍스처 해상도가 낮고, 표면 디테일이 부족하며 흐릿함. 금속 재질 표현이 부족함.",
            "score": 4
        },
        "shape_accuracy": {
            "description": "기하학적 오류는 거의 없고, 형태는 깔끔한 편. 하지만 일부 다듬어지지 않은 부분이 있음.",
            "score": 7
        },
        "symmetry": {
            "description": "모델이 완전히 대칭적이지 않으며, 회전 중 미세한 비대칭이 관찰됨.",
            "score": 6
        },
        "overall_quality": {
            "description": "모델 자체는 기능적으로 괜찮지만, 시각적으로 완성도가 낮음. 텍스처 품질이 가장 큰 약점.",
            "score": 5
        },
        "summary": "모델의 형태는 비교적 정밀하지만, 텍스처 품질이 가장 큰 문제점.",
        "improvements": [
            "텍스처 해상도 향상",
            "더 정교한 재질 표현",
            "조명 개선"
        ]
    }

    # 총점 계산 (평균값)
    scores = [
        evaluation_result["texture_quality"]["score"],
        evaluation_result["shape_accuracy"]["score"],
        evaluation_result["symmetry"]["score"],
        evaluation_result["overall_quality"]["score"]
    ]
    total_score = round(sum(scores) / len(scores), 2)  # 평균 계산 후 소수점 2자리로 반올림

    # 총점 추가
    evaluation_result["total_score"] = total_score

    return evaluation_result
