FROM python:3.11-slim
WORKDIR /app


RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# requirements.txt만 먼저 복사하여 의존성 설치 (Docker 캐싱 활용)
COPY ./src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 전체 소스코드를 /app으로 복사
COPY ./src /app

# uvicorn을 실행할 때, main.py가 /app에 있으므로 모듈 경로는 "main:app"이어야 함
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["sleep", "3600"]
