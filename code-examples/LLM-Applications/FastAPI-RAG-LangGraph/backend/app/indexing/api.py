from fastapi import APIRouter, Request, HTTPException
from app.indexing.indexer import Indexer
from app.indexing.schemas import IndexingRequest
import logging

router = APIRouter()

@router.post("/index")
async def index_documents(indexing_request: IndexingRequest):
    """
    Index documents from a given URL into the knowledge base.
    
    This endpoint processes documents by:
    1. Parsing website content (currently uses mock CSV data)
    2. Generating AI summaries for each document
    3. Creating vector embeddings for semantic search
    4. Storing everything in ChromaDB for retrieval
    
    Args:
        indexing_request (IndexingRequest): Contains URL and user information
        
    Returns:
        dict: Success status and confirmation message
        
    Raises:
        HTTPException: 500 error if indexing process fails
        
    Example:
        ```
        POST /index
        {
            "url": "https://example.com",
            "user": "admin"
        }
        ```
    """
    try:
        # Initialize indexer with URL as collection identifier
        indexer = Indexer(indexing_request.url)
        
        # Process and index the documents
        await indexer.index_data(indexing_request.url)
        
        return {
            "status": "success", 
            "message": "Documents have been successfully indexed and are ready for search"
        }

    except Exception as e:
        logging.error(f"Indexing failed for URL {indexing_request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")





