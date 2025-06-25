from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    user_id: int


class ChatResponse(BaseModel):
    response: str