from app.chat.chat_agent.state import ChatAgentState
from pydantic import BaseModel, Field
from app.openai_connect import async_openai_client
import logging


SYSTEM_PROMPT = """
You are the answer-generator in a Retrieval-Augmented Generation (RAG) pipeline.

Inputs you will receive on every call
• **chat_history** - the full ordered list of prior messages.
  - The most recent user turn(s) contain the question you must answer.
• **documents** - an array of JSON objects, each with
    • `content` : the text chunk retrieved from the website-backed vector store  
    • `url`     : the canonical URL of the web page the chunk came from  
    • `title`   : a short human-readable title (optional, for your reference)

Your tasks
1. **Understand the question** by focusing on the latest user turn(s) in *chat_history*.
2. **Ground your answer** in the supplied *documents* whenever they contain relevant facts.
3. **Write the next assistant reply** that best satisfies the user, following these rules:

   • **Cite every fact you take from a document with its `url` in square brackets.**  
     - Example:  
       “Daily backups are retained for 30 days [https://example.com/backups-policy].”  
     - If a sentence uses information from multiple documents, list the URLs
       comma-separated inside one pair of brackets:  
       “… as detailed in the incident report and the post-mortem
       [https://incidents.example.com/2024-05-outage, https://example.com/post-mortem].”

   • Paraphrase; quote at most 50 words from any single document.

   • If none of the documents support the needed information:  
     - answer from your general knowledge **without** citations, **or**  
     - explain that the information is unavailable.

   • Keep style consistent with previous assistant messages (concise,
     technically precise, markdown allowed).

   • **Do not** fabricate citations or contradict the supplied documents.

Return **only** the assistant's textual reply—no extra JSON or metadata.
"""


class GeneratedAnswer(BaseModel):
    """
    Output schema for the generator node.

    • `answer`    - the full assistant reply shown to the user.  
                    It may contain markdown and inline citations in square
                    brackets that reference the supplied URLs.  
    """
    answer: str = Field(
        ...,
        description="Assistant reply grounded in the retrieved documents."
    )


class Generator:
    """
    Generates AI responses using retrieved documents and conversation context.
    
    This class creates contextually appropriate responses by combining:
    1. Conversation history for context
    2. Retrieved document content for factual grounding
    3. Previous evaluation feedback for iterative improvement
    """

    async def generate(self, state: ChatAgentState) -> str:
        """
        Generate an AI response based on conversation context and retrieved documents.
        
        Args:
            state (ChatAgentState): Current conversation state with context and documents
            
        Returns:
            str: Generated response text with inline citations
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])
        documents = '\n'.join([doc.model_dump_json(indent=2) for doc in state.retrieved_documents])
        messages.append({
            "role": "user", 
            "content": f"###  Documents  ###\n\n{documents}"
        })

        if state.generation_evaluation and state.generation_evaluation.feedback:
            messages.append({
                "role": "user", 
                "content": f"###  Feedback about previous answer  ###\n\n{state.generation_evaluation.feedback}"
            })

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=GeneratedAnswer,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed.answer
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the response generation step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with generated response
        """
        answer = await self.generate(state)
        state.generation = answer
        state.generation_iterations += 1
        state.generation_evaluation = None  # Reset evaluation for new generation
        return state
    

generator = Generator()


