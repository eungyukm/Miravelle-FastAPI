from fastapi import HTTPException, APIRouter
from langchain_openai import OpenAI, ChatOpenAI

from schemas.llm_schemas import TextInput

router = APIRouter()

# LLM 및 Chat 모델 객체 생성
llm = OpenAI()
chat_model = ChatOpenAI()

@router.post("/llm")
async def call_llm(input_data: TextInput):
    try:
        response = llm.invoke(input_data.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")