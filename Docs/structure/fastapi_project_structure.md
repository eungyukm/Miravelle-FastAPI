
# FastAPI 프로젝트 구조 가이드

FastAPI 프로젝트를 효율적으로 관리하고 유지보수하기 위한 디렉토리 및 파일 구조 가이드입니다.

---

## 📂 기본 구조 예시

```
📦 project-root
├── 📂 app
│   ├── 📂 api
│   │   ├── 📂 v1
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   └── auth.py
│   ├── 📂 core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── 📂 models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── post.py
│   ├── 📂 schemas
│   │   ├── __init__.py
│   │   ├── user_schemas.py
│   │   └── post_schemas.py
│   ├── 📂 services
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── post_service.py
│   ├── 📂 crud
│   │   ├── __init__.py
│   │   ├── user_crud.py
│   │   └── post_crud.py
│   ├── 📂 utils
│   │   ├── __init__.py
│   │   └── password.py
│   ├── main.py
│   └── dependencies.py
├── .env
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📂 디렉토리 및 파일 설명

### 1. `api`
- `v1` 버전별 API 관리
- 라우터 및 엔드포인트 정의

### 2. `core`
- 애플리케이션 설정 (`config.py`)  
- 보안 설정 (`security.py`)  

### 3. `models`
- SQLAlchemy 모델 정의

### 4. `schemas`
- schema는 **접미사 `schemas`를 반드시 추가**

### 5. `services`
- 서비스 레이어에서 비즈니스 로직 처리  
- 외부 API 호출
- 파일 처리 및 변환
- 모델 학습, 예측

### 6. `crud`
- 데이터베이스에서의 CRUD 작업 정의  

### 7. `utils`
- 공통 유틸리티 함수 정의  

### 8. `main.py`
- FastAPI 애플리케이션 초기화 및 설정  

### 9. `dependencies.py`
- 의존성 주입 설정  

---

## 코드 작성 규칙
1. **폴더 및 파일명 컨벤션**  
   - 하위 구조는 소문자로 작성  
   - 언더스코어(`_`) 사용  

2. **라우터 구조 예시**
```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_users():
    return {"message": "User list"}
```

3. **요청(Request) 및 응답(Response) 스키마**
- `request`는 `user_schemas.py`에 작성  
- `response`는 `user_schemas.py`에 작성  

4. **비즈니스 로직과 DB 작업 분리**
   - `services`와 `crud`는 역할을 분리하고 명확하게 관리  

---

## 이외에 사항
**API 버전 관리**  
- `api/v1/` 경로에서 버전 관리를 명확히 하여 확장성 유지  

**의존성 주입 관리**  
- `dependencies.py`에서 공통적인 의존성 관리 (예: 데이터베이스 세션, 보안 설정 등)  

**로깅 및 예외 처리**  
- `core/logging.py`를 작성해 통합 로깅 관리  
- `exception_handler.py` 작성해 글로벌 예외 처리  

**환경 변수 설정**  
- `.env`에서 보안 정보 및 환경 변수 관리  
- `core/config.py`에서 `.env` 설정 로드  

