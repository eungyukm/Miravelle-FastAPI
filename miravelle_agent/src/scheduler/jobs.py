import asyncio
import httpx
import logging

logger = logging.getLogger(__name__)

async def sample_job():
    print("스케줄러 작업 실행 중")
    await asyncio.sleep(0)

evaluate_url = "http://127.0.0.1:8000/v1/evaluate-and-save/"

is_stop_image_evaluation = False

async def call_internal_api():
    global is_stop_image_evaluation

    # 작업 중단 상태일 경우 실행 안 함
    if is_stop_image_evaluation:
        logger.warning("이미지 평가 중단 상태. 작업 실행 안 함.")
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(evaluate_url)
            logger.info(f"내부 API 호출 완료: 상태 코드 = {response.status_code}, 응답 = {response.text}")

            # 상태 코드가 500 이상일 경우 조기 종료 (스케줄러는 유지)
            if response.status_code >= 500:
                logger.warning(f"상태 코드 {response.status_code}: 작업 조기 종료")
                is_stop_image_evaluation = True
                return

    except Exception as e:
        logger.error(f"내부 API 호출 중 오류 발생: {str(e)}")
        # 오류 발생 시에도 스케줄러 유지
        return