from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.chat.api import router as chat_router 
from app.indexing.api import router as indexing_router
from app.core.db import create_tables
from app.chat.models import User, Message  # Import models to register them


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_tables()
    yield


app = FastAPI(
    debug=True,
    title="RAG App",
    lifespan=lifespan,
)

app.include_router(indexing_router, prefix="/indexing")
app.include_router(chat_router, prefix="/chat")
