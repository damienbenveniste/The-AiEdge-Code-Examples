from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from app.core.db import Base

class User(Base):
    """
    User model representing registered users in the system.
    
    Attributes:
        id (int): Primary key identifier
        username (str): Unique username for the user
        messages (List[Message]): All messages associated with this user
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    messages = relationship("Message", back_populates="user")

class Message(Base):
    """
    Message model representing chat messages in conversations.
    
    Attributes:
        id (int): Primary key identifier
        user_id (int): Foreign key reference to the user who sent the message
        message (str): The actual message content
        type (str): Message type ('user' or 'assistant')
        timestamp (datetime): When the message was created
        user (User): The user who sent this message
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(String)
    type = Column(String)  # 'user' or 'assistant'
    timestamp = Column(DateTime, default=datetime.now)

    user = relationship("User", back_populates="messages")