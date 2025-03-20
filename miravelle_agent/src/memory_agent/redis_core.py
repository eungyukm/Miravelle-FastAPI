from fastapi import HTTPException, APIRouter
import redis

rotuer = APIRouter()

redis_client = redis.Redis(host='redis_service', port=6379, decode_responses=True)

@rotuer.get("/")
async def read_root():
    return {"message": "Hello, FastAPI with Redis!"}

@rotuer.post("/set/{key}/{value}")
async def set_key(key: str, value: str):
    redis_client.set(key, value)
    return {"message": f"Set {key} to {value}"}

@rotuer.get("/get/{key}")
async def get_key(key: str):
    value = redis_client.get(key)
    return {"key": key, "value": value}
