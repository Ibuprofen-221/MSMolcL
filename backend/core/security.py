from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from core.config import jwt_access_token_expire_minutes, jwt_algorithm, jwt_secret_key

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=jwt_access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, jwt_secret_key, algorithm=jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, jwt_secret_key, algorithms=[jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise ValueError("无效或过期的访问令牌") from exc
