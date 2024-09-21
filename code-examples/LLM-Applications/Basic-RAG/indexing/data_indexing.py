import os
from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
import uuid
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore



class DataIndexer:

    def __init__(self, index_name='ml-book') -> None:

        self.embeddings = OpenAIEmbeddings()
        self.index_name = index_name

        self.pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

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
        self.vectorstore = PineconeVectorStore(
            index=self.pinecone_client.Index(self.index_name),
            embedding=self.embeddings
        )

    def index_data(self, transformed_docs, batch_size=32):
        index = self.pinecone_client.Index(self.index_name)

        for i in range(0, len(transformed_docs), batch_size):
            batch = transformed_docs[i: i + batch_size]

            question_vectors = self.embeddings.embed_documents([
                question for question, _ in batch
            ])

            vector_ids = [str(uuid.uuid4()) for _ in batch]

            meta_data = [{
                'text': doc.page_content,
                'question': question,
                **doc.metadata
            } for question, doc in batch]

            vectors = [{
                'id': vector_id,
                'values': vec,
                'metadata': meta
            } for vec, vector_id, meta in zip(question_vectors, vector_ids, meta_data)]

            try: 
                response = index.upsert(vectors=vectors)
                print(response)
            except Exception as e:
                print(e)

    def get_retriever(self):
        return self.vectorstore.as_retriever()
                
            








        
        