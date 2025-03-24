from fastapi import APIRouter, HTTPException
from src.schemas.agent_schemas import AgentRequest, AgentResponse

# 라우터 생성
router = APIRouter()

class Agent:
    def __init__(self):
        # 초기화 작업 (예: 모델 로드, 설정 등)
        pass

    def process(self, prompt: str) -> str:
        # 에이전트의 실제 처리 로직 구현 (여기서는 단순 echo)
        return f"에이전트 응답: {prompt}"

# 에이전트 인스턴스 생성
agent_instance = Agent()

@router.post("/agent", response_model=AgentResponse)
async def call_agent(input_data: AgentRequest):
    try:
        result = agent_instance.process(input_data.prompt)
        return AgentResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 호출 실패: {e}")