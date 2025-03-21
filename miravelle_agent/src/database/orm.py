import os
from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# .env 파일 로드
load_dotenv()

# 환경 변수에서 값 읽기
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?sslmode=require"

# 비동기 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True)

# 세션 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

# Base 객체 생성
Base = declarative_base()

async def get_db():
    async with SessionLocal() as session:
        yield session