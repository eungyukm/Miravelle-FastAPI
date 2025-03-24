FROM python:3.11-slim
WORKDIR /app


RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# requirements.txt만 먼저 복사하여 의존성 설치 (Docker 캐싱 활용)
COPY ./src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 전체 소스코드를 /app으로 복사
COPY ./src /app
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["sleep", "3600"]
