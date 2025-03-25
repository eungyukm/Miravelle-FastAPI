from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from scheduler.jobs import sample_job

@asynccontextmanager
async def scheduler_context(app):
    scheduler = AsyncIOScheduler()
    # 예시: 10초 간격으로 sample_job 실행
    scheduler.add_job(sample_job, 'interval', seconds=10)
    scheduler.start()
    try:
        # app.state에 스케줄러 인스턴스를 저장할 수도 있음 (선택 사항)
        app.state.scheduler = scheduler
        yield
    finally:
        scheduler.shutdown()