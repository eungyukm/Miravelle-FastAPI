from fastapi import FastAPI, HTTPException
import redis

from langchain_openai import OpenAI, ChatOpenAI
from pydantic import BaseModel

from agent.agent_core import Agent
from schemas.agent_schemas import  AgentRequest, AgentResponse

app = FastAPI(title="FastAPI with Redis and LangChain")

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

# 에이전트 인스턴스 생성
agent_instance = Agent()

@app.post("/agent", response_model=AgentResponse)
async def call_agent(input_data: AgentRequest):
    try:
        result = agent_instance.process(input_data.prompt)
        return AgentResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 호출 실패: {e}")