from langchain.tools import tool
from services.image_evaluation import get_image_from_miravell

@tool(return_direct=True, args_schema=None)
def get_image_from_miravell_tool():
    """
    miravelle에서 이미지를 가져옵니다.
    - 예시: "miravelle에서 이미지 가져와"
    """
    try:
        result = get_image_from_miravell()
        return f"API 응답: {result}"
    except Exception as e:
        return f"실패: {str(e)}"