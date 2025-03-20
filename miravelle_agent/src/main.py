from fastapi import FastAPI

from agent.agent_core import rotuer as agent_router
from llm.llm_core import rotuer as llm_router
from memory_agent.redis_core import rotuer as memory_agent_router

app = FastAPI(title="FastAPI with Redis and LangChain")

app.include_router(agent_router)
app.include_router(llm_router)
app.include_router(memory_agent_router)