from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from datetime import datetime, timezone
import logging

from app.models.user import User
from app.auth.schemas import UserCreate, UserUpdatePassword
from app.auth.security import get_password_hash, verify_password
from app.core.jwt import create_access_token

logger = logging.getLogger(__name__)

class AuthService:
    """Identity Management resolving complex interactions outside the REST boundary."""
    
    @staticmethod
    def register_user(db: Session, user_in: UserCreate) -> User:
        user = db.query(User).filter(User.email == user_in.email).first()
        if user:
            logger.warning(f"Registration stopped. Email {user_in.email} already exists.")
            raise HTTPException(status_code=409, detail="Email already registered")
            
        new_user = User(
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            full_name=user_in.full_name,
            role=user_in.role
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info(f"User Registration Success: {new_user.email}")
        return new_user

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> dict:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.hashed_password):
            logger.warning(f"Unauthorized Access: Login failed for {email}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
            
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
            
        user.last_login = datetime.now(timezone.utc)
        db.commit()
        
        access_token = create_access_token(data={"sub": user.email, "role": user.role})
        logger.info(f"Login Success: {email}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    @staticmethod
    def change_password(db: Session, current_user: User, passwords: UserUpdatePassword):
        if not verify_password(passwords.current_password, current_user.hashed_password):
            logger.warning(f"Password Change Failed: {current_user.email} (Wrong current password)")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect current password")
            
        current_user.hashed_password = get_password_hash(passwords.new_password)
        db.commit()
        logger.info(f"Password Change Success: {current_user.email}")
        return {"message": "Password updated successfully"}
