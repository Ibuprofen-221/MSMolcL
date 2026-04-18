import time

import fastapi
from fastapi import APIRouter, status

from core.config import server_ip

health_router = APIRouter()


@health_router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict:
    """健康检查接口：用于确认后端服务是否正常运行。"""
    return {
        "status": "success",
        "message": "后端服务运行正常",
        "timestamp": time.time(),
        "fastapi_version": fastapi.__version__,
        "server_ip": server_ip,
    }
