from langchain.tools import tool


from services.image_evaluation import get_image_from_miravell
from fastapi import HTTPException
import httpx
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://miravelle-appservice-dsecega7bbhvefem.koreacentral-01.azurewebsites.net"

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

# @tool(return_direct=True)
# def evaluate_and_save_from_miravell_tool():
#     """
#     Miravelle 이미지를 평가하고 결과를 저장합니다.
#     - 예시: "miravell에서 이미지를 평가하고 결과를 저장해"
#     """
#     try:
#         # 평가 수행
#         logger.info("➡ Miravelle 평가 시작")
#         result = evaluate_miravelle_image()
#         logger.info(f"평가 결과: {result}")
#
#         # id와 점수를 추출
#         item_id = result.get("id")
#         clip_score = result.get("clip_score")
#         nima_score = result.get("nima_score")
#
#         if not item_id or not clip_score or not nima_score:
#             raise HTTPException(status_code=500, detail="평가 결과가 유효하지 않습니다.")
#
#         # 평균값 계산
#         evaluation_score = (clip_score + nima_score) / 2
#         logger.info(f"평균 점수 계산 완료: 평균 점수 = {evaluation_score:.2f}")
#
#         # 평가 결과 저장 API 호출
#         logger.info(f"➡ 평가 결과 저장 시작: ID={item_id}, 점수={evaluation_score:.2f}")
#         payload = {"evaluation_score": evaluation_score}
#
#         with httpx.Client(timeout=10.0) as client:
#             url = f"{BASE_URL}/api/v1/vision/image/evaluate/{item_id}/"
#             response = client.post(
#                 url,
#                 json=payload,
#                 headers={"Content-Type": "application/json"}
#             )
#
#             logger.info(f"상태 코드: {response.status_code}")
#             logger.info(f"응답 본문: {response.text}")
#
#             # 상태 코드 200 또는 201을 성공으로 처리
#             if response.status_code in (200, 201):
#                 logger.info(f"평가 결과 저장 성공: 상태 코드={response.status_code}")
#
#                 return {
#                     "status": "success",
#                     "id": item_id,
#                     "clip_score": clip_score,
#                     "nima_score": nima_score,
#                     "evaluation_score": evaluation_score,
#                     "message": "Evaluation score uploaded successfully"
#                 }
#             else:
#                 logger.error(f"저장 실패: 상태 코드 {response.status_code}, 응답: {response.text}")
#                 raise HTTPException(
#                     status_code=response.status_code,
#                     detail=f"Failed to save evaluation score: {response.text}"
#                 )
#
#     except Exception as e:
#         logger.error(f"평가 및 저장 실패: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"평가 및 저장에 실패했습니다: {str(e)}")