import pytest
from app.auth.security import get_password_hash, verify_password
from app.core.jwt import create_access_token, decode_access_token
from app.auth.schemas import UserCreate
from pydantic import ValidationError

def test_password_hashing():
    pwd = "StrongPassword123!"
    hashed = get_password_hash(pwd)
    assert verify_password(pwd, hashed)
    assert not verify_password("wrong", hashed)

def test_jwt_flow():
    token = create_access_token({"sub": "test@test.com", "role": "Farmer"})
    payload = decode_access_token(token)
    assert payload["sub"] == "test@test.com"
    assert payload["role"] == "Farmer"

def test_user_schema_validation():
    with pytest.raises(ValidationError):
        # Enforce weak password block
        UserCreate(full_name="John Doe", email="john@example.com", password="weak")
