from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.chat.models import Message, User
from typing import List, Dict


async def get_chat_history(db: AsyncSession, user_id: int, limit: int = 20) -> List[Dict[str, str]]:
    """
    Retrieve chat message history for a specific user.
    
    Args:
        db (AsyncSession): Database session
        user_id (int): ID of the user whose history to retrieve
        limit (int): Maximum number of messages to return (default: 20)
        
    Returns:
        List[Dict[str, str]]: List of messages in chat format with 'role' and 'content' keys,
                             ordered chronologically (oldest first)
    """
    query = select(Message).where(
        Message.user_id == user_id
    ).order_by(Message.timestamp.desc()).limit(limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    chat_messages = []
    for msg in reversed(messages):
        role = "user" if msg.type == "user" else "assistant"
        chat_messages.append({"role": role, "content": msg.message})
    
    return chat_messages


async def save_user_message(db: AsyncSession, user_id: int, message: str):
    """
    Save a user message to the database.
    
    Args:
        db (AsyncSession): Database session
        user_id (int): ID of the user sending the message
        message (str): The message content
        
    Returns:
        Message: The created message record
    """
    new_message = Message(
        user_id=user_id,
        message=message,
        type="user"
    )
    db.add(new_message)
    await db.commit()
    return new_message


async def save_assistant_message(db: AsyncSession, user_id: int, message: str):
    """
    Save an assistant response to the database.
    
    Args:
        db (AsyncSession): Database session
        user_id (int): ID of the user who received the response
        message (str): The assistant's response content
        
    Returns:
        Message: The created message record
    """
    new_message = Message(
        user_id=user_id,
        message=message,
        type="assistant"
    )
    db.add(new_message)
    await db.commit()
    return new_message