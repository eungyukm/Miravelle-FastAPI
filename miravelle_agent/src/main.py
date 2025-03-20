from fastapi import FastAPI, HTTPException
import redis

from langchain_openai import OpenAI, ChatOpenAI
from pydantic import BaseModel

from agent.agent_core import rotuer as agent_router

app = FastAPI(title="FastAPI with Redis and LangChain")

app.include_router(agent_router)

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI with Redis!"}

@app.post("/set/{key}/{value}")
async def set_key(key: str, value: str):
    redis_client.set(key, value)
    return {"message": f"Set {key} to {value}"}

@app.get("/get/{key}")
async def get_key(key: str):
    value = redis_client.get(key)
    return {"key": key, "value": value}

# LLM 및 Chat 모델 객체 생성
llm = OpenAI()
chat_model = ChatOpenAI()

class TextInput(BaseModel):
    text: str

@app.post("/llm")
async def call_llm(input_data: TextInput):
    try:
        response = llm.invoke(input_data.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")