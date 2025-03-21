from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
def test_route():
    return {"message": "hello"}


class VideoInput(BaseModel):
    video_url: str

class EvaluationResponse(BaseModel):
    texture_quality: int
    shape_accuracy: int
    symmetry: int
    overall_quality: int
    total_score: float
    summary: str
    raw_text: str