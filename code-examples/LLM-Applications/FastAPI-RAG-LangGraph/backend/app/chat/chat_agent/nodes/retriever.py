from app.indexing.indexer import Indexer, Data
from app.chat.chat_agent.state import ChatAgentState, RetrieveDocument
from app.openai_connect import async_openai_client
from typing import Optional, List
from pydantic import BaseModel, Field
import asyncio

import logging


SYSTEM_PROMPT = """
You are a second-stage retrieval assistant.

Inputs you will receive for **each** call  
• **chat_history** - an ordered list of all messages exchanged so far,  
  each object having `role` (“user” | “assistant” | “system”) and `content`.  
  The last one or more entries with `role == "user"` constitute the
  current **user_query** you must focus on.  
• **doc_content** - the full text of ONE document returned by the first-stage
  embedding search (it may contain noise or tangential sections).

Your tasks  
1. Decide whether *doc_content* contains information that materially helps
   answer *user_question*.  
2. If it **does** help, extract only the parts that are most relevant
   (verbatim or lightly paraphrased) and pack them into the `extracted`
   field.  
3. If it **does not** help, leave `extracted` null.  
4. Provide a one-sentence rationale and your confidence.

Guidelines  
• Favour *precision*: mark `is_relevant = true` only when the document
  supplies concrete facts, definitions, procedures, or examples that the
  question requires.  
• Keep `extracted` short — ≤ 800 characters; include just the sentences,
  bullet points, or short code blocks that the downstream RAG step should
  quote or ground itself on.  
• If unsure, set `is_relevant = false` and confidence = "low".  
• Respond with **JSON that is valid for the DocFilterResult schema**; no
  markdown, no additional keys.
"""


class DocFilterResult(BaseModel):
    """
    Result of the second-stage filtering step for a single document.
    """
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the user query, else False."
    )
    extracted: Optional[str] = Field(
        None,
        description="Key snippets or paraphrased facts taken from the document "
                    "that directly address the question; null when is_relevant is False."
    )


class Retriever:
    """
    Handles document retrieval and filtering for RAG pipeline.
    
    This class performs two-stage retrieval:
    1. Initial embedding-based search to find candidate documents
    2. LLM-based filtering to extract only relevant content
    """

    async def naive_retrieval(self, state: ChatAgentState) -> List[Data]:
        """
        Perform initial document retrieval using vector search.
        
        Args:
            state (ChatAgentState): Current conversation state with search query
            
        Returns:
            List[Data]: Raw documents from vector search
        """
        # TODO: make the collection selection dependent on user id
        indexer = Indexer('collection')
        results = await indexer.search(state.query_vector_db, max_results=20)
        return results
    
    async def filter_doc(self, state: ChatAgentState, document: Data) -> DocFilterResult:
        """
        Filter individual document for relevance using LLM.
        
        Args:
            state (ChatAgentState): Current conversation state
            document (Data): Document to evaluate for relevance
            
        Returns:
            DocFilterResult: Filtered result with relevance flag and extracted content
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])
        messages.append({"role": "user", "content": f"###  Document  ###\n\n{document.text}"})

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=DocFilterResult,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    async def filter_documents(self, state: ChatAgentState, documents: List[Data]) -> List[DocFilterResult]:
        """
        Filter multiple documents in parallel for relevance.
        
        Args:
            state (ChatAgentState): Current conversation state
            documents (List[Data]): Documents to filter
            
        Returns:
            List[DocFilterResult]: Filtered results for each document
        """
        tasks = [
            asyncio.create_task(self.filter_doc(state, doc)) 
            for doc in documents
        ]
        filters = await asyncio.gather(*tasks, return_exceptions=True)
        return filters
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the document retrieval and filtering process.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with filtered documents
        """
        documents = await self.naive_retrieval(state)
        filters = await self.filter_documents(state, documents)
        retrieved_documents = []

        # Extract only relevant document content
        for doc, filter in zip(documents, filters):
            if filter.is_relevant and filter.extracted:
                retrieved_documents.append(
                    RetrieveDocument(text=filter.extracted, url=doc.url)
                )

        state.retrieved_documents = retrieved_documents
        state.retrieval_iterations += 1
        return state
    

retriever = Retriever()


        


    


