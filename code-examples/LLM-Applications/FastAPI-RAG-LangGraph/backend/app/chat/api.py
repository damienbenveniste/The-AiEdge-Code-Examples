from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.chat.chat_agent.agent import chat_agent
from app.chat.chat_agent.state import ChatAgentState
from app.chat.schemas import ChatRequest, ChatResponse
from app.chat.cruds import get_chat_history, save_user_message, save_assistant_message
from app.core.db import get_db
import logging

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Process a chat message and generate an AI response.
    
    This endpoint handles the complete conversational flow:
    1. Saves the incoming user message to the database
    2. Retrieves conversation history for context
    3. Routes the conversation through the multi-step chat agent
    4. Saves the AI response to the database
    5. Returns the generated response
    
    The chat agent uses a sophisticated RAG (Retrieval-Augmented Generation)
    pipeline that can route between simple responses and knowledge-base
    enhanced responses based on the query complexity.
    
    Args:
        request (ChatRequest): Contains the user message and user_id
        db (AsyncSession): Database session for persistence operations
        
    Returns:
        ChatResponse: Contains the AI-generated response text
        
    Raises:
        HTTPException: 500 error if agent processing fails
        
    Example:
        ```
        POST /message
        {
            "message": "What is the pricing model?",
            "user_id": 1
        }
        ```
    """
    try:
        # Save user message first
        await save_user_message(db, request.user_id, request.message)
        
        # Retrieve chat history from database
        chat_messages = await get_chat_history(db, request.user_id)
        
        # Add current message if not already in history
        if not chat_messages or chat_messages[-1]["content"] != request.message:
            chat_messages.append({"role": "user", "content": request.message})
        
        # Initialize state
        initial_state = ChatAgentState(
            chat_messages=chat_messages,
        )
        
        # Run the agent
        result = await chat_agent.ainvoke(initial_state, debug=True)
        
        # Cast result to ChatAgentState
        final_state = ChatAgentState(**result)
        
        # Save assistant response to database
        response_text = final_state.generation or "I'm sorry, I couldn't generate a response."
        await save_assistant_message(db, request.user_id, response_text)
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        logging.error(f"Chat agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))