# Authentication & Authorization Architecture

AgroCarbonAI employs an entirely decoupled Identity layer based on robust `boto3` equivalent abstraction logic (via `SQLAlchemy` and `Python-Jose`). This strictly keeps Identity outside the AI modules.

## Security Constraints
- **Passwords**: Encrypted strictly via `bcrypt` (12 Rounds preventing side-channel attacks). Minimum 8chars + lower/upper/numeric/special reqs.
- **Tokens**: Utilizing JWT via `OAuth2PasswordBearer` mapping cleanly to React/Flutter HTTP Interceptors.

## Environment Variables
Append to `.env`:
```env
JWT_SECRET_KEY="replace-with-minimum-32-char-random-string"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES="60"
BCRYPT_ROUNDS="12"
PASSWORD_MIN_LENGTH="8"
```

## Protected Endpoints Flow
1. **Registration**: Client sends POST `/api/v1/auth/register`.
2. **Login**: Client sends POST `/api/v1/auth/login`. Server yields `Bearer`.
3. **Execution**: Client hits `POST /api/v1/analyze` attaching Header: `Authorization: Bearer <token>`.
4. **Validation**: Dependency `get_current_user` catches token, decrypts, and rejects 401 instantaneously if tampered.

## Role Based Access Control (RBAC)
When calling `/prediction/{id}`, the system queries the token's original `user_id`.
If `token.user_id != prediction.user_id`, a `403 Forbidden` response is aggressively thrown.
*Exception*: Any token sporting `"role": "Admin"` bypasses strict ownership restrictions.

## API Documentation Sync
Navigate to `127.0.0.1:8000/docs`. You will now see a **Padlock / Authorize** button at the top right natively enabling browser testing.
