from fastapi import APIRouter, HTTPException
from schemas.agent_schemas import AgentRequest, AgentResponse

# langchain
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, tool, AgentType

# schemas
from schemas.llm_schemas import CommandRequest

# tools
from .tools_core import *


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

# LangChain 설정 (단일 입력값으로 수정)
tools = [
    get_image_from_api,
    upload_to_huggingface,
    get_image_from_miravell_tool,
    evaluate_and_save_miravell,
]

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Tool 등록 상태 출력
print(f"Registered Tools: {[tool.name for tool in tools]}")

@router.post("/v1/process-command")
def process_command(request: CommandRequest):
    """
    에이전트를 통해 명령을 처리합니다.
    """
    try:
        result = agent.invoke(request.command)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))