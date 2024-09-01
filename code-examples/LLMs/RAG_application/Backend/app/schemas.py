from pydantic.v1 import BaseModel
from datetime import datetime
from typing import Optional

class UserRequest(BaseModel):
    username: str
    question: str

class RagInput(BaseModel):
    chat_history: str
    question: str
