import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# 연결 문자열 설정
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# 비동기 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True)

# 연결 테스트 함수
async def test_connection():
    try:
        async with engine.begin() as conn:
            # 수정된 부분: 문자열 쿼리를 text()로 감싸기
            result = await conn.execute(text("SELECT 1"))
            print("연결 성공! 결과:", result.all())
    except Exception as e:
        print(f"연결 실패: {e}")

# 연결 테스트 실행
if __name__ == "__main__":
    asyncio.run(test_connection())
