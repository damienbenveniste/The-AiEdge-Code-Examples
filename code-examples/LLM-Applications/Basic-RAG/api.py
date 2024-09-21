from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from retrieval.conversation_qa import RAGPipeline
from indexing.data_indexing import DataIndexer

retriever = DataIndexer().get_retriever()
rag_chain = RAGPipeline(retriever).get_chain()

app = FastAPI(
    title='RAG pipeline',
    description='a simple rag pipeline'
)


add_routes(
    app,
    ChatOpenAI(model='gpt-4o-mini'),
    path='/openai'
)

add_routes(
    app,
    rag_chain,
    path='/rag'
)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app',host='localhost', reload=True)