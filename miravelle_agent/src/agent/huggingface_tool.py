#agent/huggingface_tool.py
from langchain.tools import tool
import os
import json
import logging
from huggingface_hub import HfApi

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

@tool(return_direct=True)
def upload_to_huggingface(score, image_data):
    """
    평가 점수와 이미지를 허깅페이스 데이터셋에 업로드합니다.
    """
    try:
        api = HfApi()

        # 데이터셋 디렉토리 생성
        os.makedirs("dataset", exist_ok=True)

        # 이미지 파일 저장
        image_file = "dataset/evaluated_image.jpg"
        with open(image_file, "wb") as img_f:
            img_f.write(image_data)

        # 점수 데이터 저장
        data = {"score": score}
        dataset_file = "dataset/nima_scores.jsonl"
        with open(dataset_file, "a") as f:
            f.write(json.dumps(data) + "\n")

        # 허깅페이스에 파일 업로드
        api.upload_file(
            path_or_fileobj=dataset_file,
            path_in_repo="nima_scores.jsonl",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HUGGINGFACE_TOKEN
        )

        api.upload_file(
            path_or_fileobj=image_file,
            path_in_repo="evaluated_image.jpg",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HUGGINGFACE_TOKEN
        )

        logger.info(f"점수 {score}와 이미지가 {DATASET_REPO}에 업로드되었습니다.")
        return f"점수 {score}와 이미지가 {DATASET_REPO}에 업로드되었습니다."

    except Exception as e:
        logger.error(f"허깅페이스 업로드 실패: {str(e)}")
        raise Exception(f"허깅페이스 업로드 실패: {str(e)}")
