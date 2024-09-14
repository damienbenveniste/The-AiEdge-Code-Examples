from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./code_data/langchain_repo/",
    branch="master",
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=10000, chunk_overlap=100
)

docs = loader.load()
docs = [doc for doc in docs if len(doc.page_content) < 50000]

vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_langchain_db", 
)

vectorstore.add_documents(documents=docs)
retriever = vectorstore.as_retriever(k=5)