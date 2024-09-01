from sqlalchemy.orm import Session
import models, schemas


def get_or_create_user(db: Session, username: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        user = models.User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def add_message(db: Session, message: schemas.MessageBase, username: str):
    user = get_or_create_user(db, username)
    message = models.Message(**message.dict())
    message.user = user
    db.add(message)
    db.commit()
    db.refresh(message)
    return message

def get_user_chat_history(db: Session, username: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return []
    return user.messages