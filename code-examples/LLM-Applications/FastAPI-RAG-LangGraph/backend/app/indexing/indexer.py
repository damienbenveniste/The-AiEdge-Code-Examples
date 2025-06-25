import chromadb
from app.openai_connect import async_openai_client
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import os


SYSTEM_PROMPT = """
You are an expert summarizer.
Summarize the web page provided by the user in **no more than two concise, descriptive sentences**.
"""

class Summary(BaseModel):
    """Schema for a one- or two-sentence web-page summary."""
    summary: str = Field(
        ...,
        description="A concise description of the page (≤ 2 sentences)."
    )


class Data(BaseModel):
    text: str
    url: str
    summary: Optional[str] = None


class Indexer:
    """
    Manages document indexing and retrieval using ChromaDB vector database.
    
    This class handles:
    1. Document parsing and summarization
    2. Embedding generation for semantic search
    3. Vector storage and retrieval operations
    """

    def __init__(self, collection_name) -> None:
        """
        Initialize the indexer with a specific collection.
        
        Args:
            collection_name (str): Name of the ChromaDB collection to use
        """
        # TODO: capture `collection_name` in DB to dynamically adapt to user data
        self.collection_name = 'collection'
        client_db = chromadb.PersistentClient()
        self.index = client_db.get_or_create_collection(name=self.collection_name)

    def mock_parse_website(self, url: str) -> List[Data]:
        """
        Parse website data from CSV file (mock implementation).
        
        Args:
            url (str): URL parameter (currently unused in mock)
            
        Returns:
            List[Data]: Parsed document data from CSV
        """
        FILE_PATH = os.path.join(os.path.dirname(__file__), 'parsed_website.csv')
        df = pd.read_csv(FILE_PATH)
        # remove duplicates and `page not found`
        df = df.loc[
            ~df['markdown'].str.contains(
                "page not found", case=False, na=False
            ), ["crawl/loadedUrl", 'text']
        ].drop_duplicates(subset=['crawl/loadedUrl']).dropna().copy()

        data = list(df.T.to_dict().values())
        data = [Data(text=d['text'].strip(), url=d['crawl/loadedUrl'].strip()) for d in data ]
        return data
    
    async def summarize_text(self, text: str) -> str:
        """
        Generate a concise summary of document text using AI.
        
        Args:
            text (str): Document text to summarize
            
        Returns:
            str: AI-generated summary (≤ 2 sentences)
        """
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Webpage: {text}"}
        ]

        try: 
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-nano',
                input=messages,
                temperature=0.1,
                text_format=Summary,
            )
        except Exception as e:
            print(e)

        return response.output_parsed.summary
    
    async def summarize_all(self, data: List[Data]) -> List[Data]:
        """
        Generate summaries for all documents in parallel.
        
        Args:
            data (List[Data]): List of documents to summarize
            
        Returns:
            List[Data]: Documents with generated summaries added
        """
        tasks = [
            asyncio.create_task(self.summarize_text(page.text)) 
            for page in data
        ]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        for page, summary in zip(data, summaries):
            page.summary = summary.strip()

        return data
    
    async def embed_data(self,  data: List[Data]) -> List[float]:
        """
        Generate vector embeddings for document summaries.
        
        Args:
            data (List[Data]): Documents with summaries to embed
            
        Returns:
            List[float]: Vector embeddings for semantic search
        """
        response = await async_openai_client.embeddings.create(
            input=[page.summary for page in data],
            model="text-embedding-3-small"
        )
        embeddings = [res.embedding for res in response.data]
        return embeddings
    
    async def index_data(self, url: str) -> None:
        """
        Index website data by parsing, summarizing, and storing embeddings.
        
        Args:
            url (str): URL to index (currently passed to mock parser)
        """
        # Skip if collection already has data
        if self.index.count() != 0:
            return
        
        data = self.mock_parse_website(url)
        data = await self.summarize_all(data)
        embeddings = await self.embed_data(data)

        self.index.add(
            documents=[d.model_dump_json() for d in data],
            embeddings=embeddings,
            ids=[d.url for d in data]
        )

    async def search(self, query: str, max_results: int = 15) -> List[Data]:
        """
        Perform semantic search over indexed documents.
        
        Args:
            query (str): Search query text
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Data]: Most relevant documents ranked by similarity
        """
        response = await async_openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        results = self.index.query(
            query_embeddings=[response.data[0].embedding],
            n_results=max_results,
            include=["documents"]
        )

        docs = results["documents"][0]  # ChromaDB returns nested list
        data = [Data.model_validate_json(doc) for doc in docs]
        return data


# Quick test script
if __name__ == "__main__":
    import asyncio
    
    async def test_indexer():
        indexer = Indexer("collection")
        
        # Check if data exists
        count = indexer.index.count()
        print(f"Total documents in collection: {count}")
        
        if count > 0:
            # Test search
            results = await indexer.search("test", max_results=3)
            print(f"\nSearch results for 'test': {len(results)} found")
            for i, result in enumerate(results):
                print(f"{i+1}. URL: {result.url}")
                print(f"Summary: {result.summary}")
                print()
        else:
            print("No data found in collection")
    
    asyncio.run(test_indexer())
