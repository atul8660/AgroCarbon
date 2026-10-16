from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.auth.schemas import UserCreate, UserResponse, Token, UserUpdatePassword
from app.auth.service import AuthService
from app.auth.dependencies import get_current_user, get_db
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Registers new user mapping complex password requirements."""
    return AuthService.register_user(db, user_in)

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 password bearer compliant login generating strict JWT tokens."""
    return AuthService.authenticate_user(db, form_data.username, form_data.password)

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Decodes JWT internally returning contextual caller identity."""
    return current_user

@router.put("/change-password")
def change_password(passwords: UserUpdatePassword, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Strictly modifies password logic forcing current validation."""
    return AuthService.change_password(db, current_user, passwords)
    
@router.post("/logout")
def logout():
    """Stateless JWT dictates logout occurs on Client Side via securely dropping the token."""
    return {"message": "Stateless logout successful. Client must drop token."}
