# Miravelle-FastAPI

# 요구 사항
Docker와 Docker Compose가 설치되어 있어야 합니다.

# 실행
Docker Compose를 사용해 FaseAPI와 Redis를 동시에 실행합니다.
```
docker-compose up --build
```

# 종료
컨테이너와 네트워크를 정리합니다.
```
docker-compose down
```

# Docker 컨테이너 재빌드
```
# 기존 컨테이너 중지 및 삭제
docker-compose down

# 이미지 캐시 무시하고 새로 빌드
docker-compose up --build --force-recreate
```