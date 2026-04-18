from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.db import get_db
from core.security import decode_access_token
from models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


unauthorized_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="认证失败，请重新登录",
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise unauthorized_exception

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError:
        raise unauthorized_exception

    username = str(payload.get("sub") or "").strip()
    if not username:
        raise unauthorized_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise unauthorized_exception

    return user
