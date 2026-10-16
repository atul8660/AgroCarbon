import re
from pydantic import BaseModel, EmailStr, Field, field_validator
from pydantic import ConfigDict

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = "Farmer"

    @field_validator('password')
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        """Enforces absolute minimal complexity for production database entries avoiding brute force dictionaries."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

class UserUpdatePassword(BaseModel):
    current_password: str
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v): raise ValueError("Password requires uppercase letter.")
        if not re.search(r"[a-z]", v): raise ValueError("Password requires lowercase letter.")
        if not re.search(r"[0-9]", v): raise ValueError("Password requires number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v): raise ValueError("Password requires special character.")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str | None
    role: str
    is_active: bool
    
    model_config = ConfigDict(from_attributes=True)
