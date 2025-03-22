from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class CommandRequest(BaseModel):
    command: str