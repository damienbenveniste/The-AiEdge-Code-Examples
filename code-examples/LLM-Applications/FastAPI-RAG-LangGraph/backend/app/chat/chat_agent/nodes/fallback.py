from app.chat.chat_agent.state import ChatAgentState
from pydantic import BaseModel, Field
from app.openai_connect import async_openai_client
from typing import Optional
import logging


SYSTEM_PROMPT = """
You are a fallback assistant.

Situation  
• The pipeline could not find sufficient information in the retrieved
  documents to answer the user's latest question.

Your job  
1. Send a short, polite apology.  
2. Briefly explain that the necessary information was not present in the
   available context.  
3. Offer one helpful next step (e.g. rephrase the question, provide more
   details, or try a different topic).

Guidelines  
• Keep the tone friendly and professional.  
• Do **not** invent an answer or cite any sources.  
• Limit the entire reply to ≤ 3 short sentences.  
• Return JSON that conforms to the ApologyReply schema—no extra keys, no
  markdown.
"""


class ApologyReply(BaseModel):
    """
    Fallback response when no grounded answer is possible.
    """
    apology: str = Field(
        ...,
        description="A brief apology acknowledging the inability to answer."
    )
    suggestion: Optional[str] = Field(
        None,
        description="Optional next-step guidance for the user (rephrase, add details, etc.)."
    )
    

class FallBack:
    """
    Provides fallback responses when the main pipeline fails to generate answers.
    
    This class handles edge cases where:
    1. No relevant documents are found in the knowledge base
    2. Generation quality is too poor after multiple iterations
    3. System errors prevent normal response generation
    """

    async def fallback_answer(self, state: ChatAgentState) -> ApologyReply:
        """
        Generate a polite fallback response when the main pipeline fails.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ApologyReply: Structured apology with optional guidance
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=ApologyReply,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the fallback response generation step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with fallback response
        """
        reply = await self.fallback_answer(state)
        state.generation = reply.apology + (" " + reply.suggestion if reply.suggestion else "")
        return state
    

fallback = FallBack()