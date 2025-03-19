from fastapi import FastAPI
import redis

app = FastAPI()

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI with Redis!"}

@app.post("/set/{key}/{value}")
async def set_key(key: str, value: str):
    redis_client.set(key, value)
    return {"message": f"Set {key} to {value}"}

@app.get("/get/{key}")
async def get_key(key: str):
    value = redis_client.get(key)
    return {"key": key, "value": value}