from fastapi import FastAPI

from agent.agent_core import router as agent_router
from llm.llm_core import router as llm_router
from memory_agent.redis_core import router as memory_agent_router
from api.v1.gemini import router as gemini_router
from api.v1.image_evaluator import router as image_router
from api.v1.model_selector import router as model_router
from api.v1.image_evaluation_agent import router as image_evaluation_agent_router
from api.v1.nima_predictor import router as nima_router

# scheduler
from scheduler.scheduler_core import scheduler_context


# FastAPI 애플리케이션 생성 및 lifespan 설정
app = FastAPI(title="FastAPI with Redis and LangChain, CLIP, and NIMA", lifespan=scheduler_context)

app.include_router(agent_router)
app.include_router(llm_router)
app.include_router(memory_agent_router)
app.include_router(gemini_router, prefix="/model")
app.include_router(image_router, prefix="/image")
app.include_router(model_router, prefix="/model-selector")
app.include_router(image_evaluation_agent_router)
app.include_router(nima_router, prefix="/nima")