import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.auth import get_current_user
from core.response import success_response
from models.user import User
from util.file_utils import get_user_file_path_readonly, resolve_user_data_dir

statas_router = APIRouter(prefix="/api", tags=["statas"])

ALLOWED_STATAS_FILENAMES = {"statas.json", "statas_advanced.json"}


def _resolve_statas_filename(result_type: str) -> str:
    if result_type == "normal":
        return "statas.json"
    if result_type == "advanced":
        return "statas_advanced.json"
    raise HTTPException(status_code=400, detail="result_type 仅支持 normal/advanced")


def _validate_path(target: Path, allowed_root: Path) -> Path:
    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="statas文件不存在") from None

    if not str(resolved).startswith(str(allowed_root.resolve())):
        raise HTTPException(status_code=400, detail="仅允许读取当前用户目录下的statas文件") from None

    if resolved.name not in ALLOWED_STATAS_FILENAMES:
        raise HTTPException(status_code=400, detail="仅允许读取statas.json或statas_advanced.json")

    return resolved


@statas_router.get("/statas", status_code=status.HTTP_200_OK)
async def get_statas(
    task_id: str | None = Query(default=None, description="任务ID（优先使用）"),
    path: str | None = Query(default=None, description="statas文件绝对路径（仅允许当前用户目录）"),
    task_space: str = Query(default="normal", description="任务空间：normal/advanced（advanced兼容映射到normal目录）"),
    result_type: str = Query(default="normal", description="结果类型：normal/advanced"),
    current_user: User = Depends(get_current_user),
) -> dict:
    """返回当前用户目录中的 statas 内容，仅读不写。"""
    if task_space not in {"normal", "advanced"}:
        raise HTTPException(status_code=400, detail="task_space 仅支持 normal/advanced")

    target_filename = _resolve_statas_filename(result_type)
    user_root = resolve_user_data_dir(current_user.data_path)

    if task_id:
        target_path = get_user_file_path_readonly(current_user.data_path, task_id, target_filename, storage="normal")
    elif path:
        target_path = Path(path)
    else:
        raise HTTPException(status_code=400, detail="task_id 与 path 不能同时为空")

    resolved = _validate_path(target_path, user_root)

    try:
        content = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="读取statas失败") from None

    return success_response(data=content, message="statas读取成功")
