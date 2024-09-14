import os
import uuid
from pathlib import Path
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class DataIndexer:

    def __init__(self, index_name='langchain-repo') -> None:
        self.embedding_client = OpenAIEmbeddings()

        self.index_name = index_name
        self.pinecone_client = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

        if index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            ) 

        self.index = self.pinecone_client.Index(self.index_name)

    def index_data(self, docs, batch_size=32):

        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            values = self.embedding_client.embed_documents([
                doc.page_content for doc in batch
            ])
            vector_ids = [str(uuid.uuid4()) for _ in batch]
 
            metadatas = [{
                'text': doc.page_content,
                **doc.metadata
            } for doc in batch]

            vectors = [{
                'id': vector_id,
                'values': value,
                'metadata': metadata
            } for vector_id, value, metadata in zip(vector_ids, values, metadatas)]

            try: 
                upsert_response = self.index.upsert(vectors=vectors)
                print(upsert_response)
            except Exception as e:
                print(e)

    def search(self, text_query, top_k=5):

        vector = self.embedding_client.embed_query(text_query)
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )

        docs = []
        for res in result["matches"]:
            metadata = res["metadata"]
            if 'text' in metadata:
                text = metadata.pop('text')
                docs.append(text)
        return docs
    

if __name__ == '__main__':

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
    docs = [doc for doc in docs if doc.metadata['file_type'] in ['.py', '.md']]
    docs = [doc for doc in docs if len(doc.page_content) < 50000]
    docs = python_splitter.split_documents(docs)
    for doc in docs:
        doc.page_content = '# {}\n\n'.format(doc.metadata['source']) + doc.page_content

    indexer = DataIndexer()
    indexer.index_data(docs)


