
from pydantic import BaseModel


class IndexingRequest(BaseModel):
    url: str
    user: str