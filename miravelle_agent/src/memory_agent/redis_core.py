from fastapi import APIRouter
import redis
import json
import os

router = APIRouter()

redis_client = redis.Redis(host='redis_service', port=6379, decode_responses=True)


def load_commands_to_redis():
    # 절대 경로로 commands.json 로드
    file_path = os.path.join("/app", "commands.json")
    with open(file_path, "r") as file:
        commands = json.load(file)
        for key, value in commands.items():
            redis_client.set(key, value)
            print(f"{key} 명령어 저장 완료!")

@router.post(("/startup"))
def startup():
    load_commands_to_redis()

@router.get("/list-commands/")
def list_commands():
    keys = redis_client.keys('command_*')

    # key가 bytes 타입인지 확인하고 decode 처리
    commands = {
        key.decode('utf-8') if isinstance(key, bytes) else key:
            redis_client.get(key).decode('utf-8') if isinstance(redis_client.get(key), bytes) else redis_client.get(key)
        for key in keys
    }
    return {"commands": commands}

@router.post("/set/{key}/{value}")
async def set_key(key: str, value: str):
    redis_client.set(key, value)
    return {"message": f"Set {key} to {value}"}

@router.get("/get/{key}")
async def get_key(key: str):
    value = redis_client.get(key)
    return {"key": key, "value": value}
