from fastapi import HTTPException, APIRouter
import redis

router = APIRouter()

redis_client = redis.Redis(host='redis_service', port=6379, decode_responses=True)

@router.get("/")
async def read_root():
    return {"message": "Hello, FastAPI with Redis!"}

@router.post("/set/{key}/{value}")
async def set_key(key: str, value: str):
    redis_client.set(key, value)
    return {"message": f"Set {key} to {value}"}

@router.get("/get/{key}")
async def get_key(key: str):
    value = redis_client.get(key)
    return {"key": key, "value": value}
