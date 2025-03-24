from pydantic import BaseModel

class EvaluationRequest(BaseModel):
    evaluation_score: float