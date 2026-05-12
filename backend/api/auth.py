import math
import shutil

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from core.auth_guard import LoginGuard
from core.config import rate_limit_auth_login_rule, rate_limit_auth_register_rule
from core.db import get_db
from core.rate_limit import get_request_client_ip, limiter
from core.response import success_response
from core.security import create_access_token, get_password_hash, verify_password
from models.user import User
from schemas.auth import LoginRequest, RegisterRequest
from util.file_utils import init_user_data_layout, to_user_data_relative_path

auth_router = APIRouter(tags=["认证"])
login_guard = LoginGuard()


@auth_router.post("/register", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_auth_register_rule)
async def register_api(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    _ = request
    username = (payload.username or "").strip()
    password = payload.password or ""

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    exists = db.query(User).filter(User.username == username).first()
    if exists is not None:
        raise HTTPException(status_code=400, detail="用户已存在")

    user_rel_path = to_user_data_relative_path(username)

    try:
        user_dir = init_user_data_layout(user_rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=500, detail="初始化用户目录失败") from None

    new_user = User(
        username=username,
        hashed_password=get_password_hash(password),
        data_path=user_rel_path,
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception:
        db.rollback()
        shutil.rmtree(user_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="注册失败") from None

    return success_response(
        message="注册成功",
        data={"id": new_user.id, "username": new_user.username},
    )


@auth_router.post("/login", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_auth_login_rule)
async def login_api(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    username = (payload.username or "").strip()
    password = payload.password or ""
    client_ip = get_request_client_ip(request)

    blocked_seconds = login_guard.get_blocked_seconds(client_ip, username)
    if blocked_seconds > 0:
        wait_minutes = max(1, math.ceil(blocked_seconds / 60))
        raise HTTPException(status_code=429, detail=f"登录失败次数过多，请 {wait_minutes} 分钟后重试")

    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.hashed_password):
        wait_seconds = login_guard.record_failure(client_ip, username)
        if wait_seconds > 0:
            wait_minutes = max(1, math.ceil(wait_seconds / 60))
            raise HTTPException(status_code=429, detail=f"登录失败次数过多，请 {wait_minutes} 分钟后重试")
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    login_guard.reset(client_ip, username)
    access_token = create_access_token(subject=user.username)
    return success_response(
        message="登录成功",
        data={"access_token": access_token, "token_type": "bearer", "username": user.username},
    )
