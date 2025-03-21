from pydantic import BaseModel

class VideoInput(BaseModel):
    video_url: str

class EvaluationResponse(BaseModel):
    evaluation: str