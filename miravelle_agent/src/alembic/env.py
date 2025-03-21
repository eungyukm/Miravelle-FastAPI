from logging.config import fileConfig
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import pool
from alembic import context
from database.orm import Base
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Alembic 설정
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# DB 연결 정보 설정
DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# FastAPI에서 사용할 메타데이터
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """오프라인 모드에서 마이그레이션 실행"""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        version_table_schema="fastapi_schema"
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online() -> None:
    """온라인 모드에서 비동기 마이그레이션 실행"""
    connectable: AsyncEngine = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(
            lambda conn: context.configure(
                connection=conn,
                target_metadata=target_metadata,
                include_schemas=True,
                version_table_schema="fastapi_schema"
            )
        )

        async with connection.begin():
            # 인자 수정됨
            await connection.run_sync(lambda conn: context.run_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio
    asyncio.run(run_migrations_online())
