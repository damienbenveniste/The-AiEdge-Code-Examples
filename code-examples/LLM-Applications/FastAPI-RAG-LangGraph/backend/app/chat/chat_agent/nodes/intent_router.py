from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from app.openai_connect import async_openai_client
from app.indexing.indexer import Indexer
from app.chat.chat_agent.state import ChatAgentState
import logging


SYSTEM_PROMPT = """
You are a routing assistant operating inside a multi-turn chat.

Inputs (JSON) you will receive on every call  
• **knowledge_base_samples** - ten short summaries, each describing the contents of a single web-page that lives inside the same website.  
  - Treat them as a representative sample of *all* documents stored in the vector database.  
  - Infer from them the subject-matter scope and level of detail the database can provide.
• **chat_history** - an ordered list of all messages exchanged so far,  
  each object having `role` (“user” | “assistant” | “system”) and `content`.  
  The last one or more entries with `role == "user"` constitute the
  current **user_query** you must route.

Your tasks  
1. Extract the current **user_query** (the most recent user turn).  
2. Decide whether answering **user_query** **requires retrieving additional
   context** from the website-backed vector database (Retrieval-Augmented
   Generation, “RAG”) or can be answered directly from your own pretrained
   knowledge.  
3. If retrieval **is** required, craft a concise search string
   (3-12 keywords or phrases, proper nouns welcome) that will surface the
   most relevant pages.

Decision guidelines  
• Use **needs_rag = false** only when the question is clearly general
  knowledge, definitional, or solvable with a brief calculation and the
  website is unlikely to add meaningful detail.  
• Use **needs_rag = true** when the question asks for website-specific
  facts, statistics, procedures, or entities, or when prior turns show the
  user seeking such data.  
• If uncertain, default to **needs_rag = true**.

Focus rules  
• Prioritize the **latest user turn(s)**; scan earlier history only to
  disambiguate pronouns or references.  
• Ignore your own previous replies except for needed context; do not let
  them bias whether RAG is required.  
• The conversation may contain unrelated digressions—base your judgement
  solely on information relevant to the current user_query.
"""


class RouterDecision(BaseModel):
    """
    Decision emitted by the routing LLM.
    """
    needs_rag: bool = Field(
        ...,
        description="True if the query should be answered with RAG; False otherwise."
    )
    query_vector_db: Optional[str] = Field(
        None,
        description="Search string for the vector DB when needs_rag is True; null otherwise."
    )


class IntentRouter:
    """
    Routes user queries to determine if RAG (Retrieval-Augmented Generation) is needed.
    
    This class analyzes the user's query and conversation history to decide whether
    the question can be answered directly or requires retrieving additional context
    from the knowledge base.
    """

    async def route(self, state: ChatAgentState) -> RouterDecision:
        """
        Determine routing decision for the user query.
        
        Args:
            state (ChatAgentState): Current conversation state with chat history
            
        Returns:
            RouterDecision: Contains needs_rag flag and optional search query
        """

        samples = await self.get_samples(state)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Knowledge_base_samples  ###\n\n{"\n".join(samples)}"},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=RouterDecision,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    async def get_samples(self, state: ChatAgentState) -> List[str]:
        """
        Get sample documents from the knowledge base for routing context.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            List[str]: Sample document summaries to help with routing decision
        """
        # TODO: dynamically change collection for specific user
        indexer = Indexer('collection')
        results = await indexer.search(state.chat_messages[-1]['content'], max_results=10)
        return [res.summary for res in results]
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the intent routing step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with routing decision
        """
        router_decision = await self.route(state)
        state.need_rag = router_decision.needs_rag
        state.query_vector_db = router_decision.query_vector_db

        return state
    

intent_router = IntentRouter()
