from langchain_core.runnables import Runnable
from langchain_core.callbacks import BaseCallbackHandler
from fastapi import FastAPI, Request, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from langserve.serialization import WellKnownLCSerializer
from typing import List
import crud, models, schemas
from database import SessionLocal, engine
from chains import rag_chain
from prompts import format_chat_history
from callbacks import LogResponseCallback


models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def generate_stream(input_data: schemas.BaseModel, runnable: Runnable, callbacks: List[BaseCallbackHandler]=[]):
    for output in runnable.stream(input_data.dict(), config={"callbacks": callbacks}): 
        data = WellKnownLCSerializer().dumps(output).decode("utf-8")
        yield {'data': data, "event": "data"} 
    yield {"event": "end"}


@app.post("/rag/stream")
async def rag_stream(request: Request, db: Session = Depends(get_db)):  
    data = await request.json()
    user_request = schemas.UserRequest(**data['input'])
    chat_history = crud.get_user_chat_history(db=db, username=user_request.username)
    message = schemas.MessageBase(message=user_request.question, type='User')
    crud.add_message(db, message, user_request.username)

    rag_input = schemas.RagInput(
        question=user_request.question,
        chat_history=format_chat_history(chat_history),
    )

    return EventSourceResponse(generate_stream(
        rag_input, 
        rag_chain,
        [LogResponseCallback(user_request, db)]
    ))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", reload=True,  port=8002)