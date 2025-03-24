from fastapi import APIRouter, UploadFile, File
from enum import Enum
from PIL import Image
import io

from src.models.predictor import predict
router = APIRouter()

class Task(str, Enum):
    texture = "texture"
    grotesque = "grotesque"
    object = "object"
    style = "style"

@router.post("/predict/{task}")
async def predict_image(task: Task, image: UploadFile = File(...)):
    contents = await image.read()
    image_data = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict(task.value, image_data)
    return {"result": result}
