from fastapi import FastAPI
from src.agent.agent_core import router as agent_router
from src.llm.llm_core import router as llm_router
from src.memory_agent.redis_core import router as memory_agent_router
from src.api.v1.gemini import router as gemini_router
from src.api.v1.image_evaluator import router as image_router
from src.api.v1.model_selector import router as model_router
from src.api.v1.image_evaluation_agent import router as image_evaluation_agent_router
from src.api.v1.image_uploader import router as upload_router


app = FastAPI()

app.include_router(agent_router)
app.include_router(llm_router)
app.include_router(memory_agent_router)
app.include_router(gemini_router, prefix="/model")
app.include_router(image_router, prefix="/image")
app.include_router(model_router, prefix="/model-selector")
app.include_router(image_evaluation_agent_router)
app.include_router(upload_router, prefix="/api/v1")