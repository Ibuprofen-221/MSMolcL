from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.auth import get_current_user
from core.response import success_response
from models.user import User
from services.sirius_batch_service import (
    SiriusBatchError,
    get_mgf_only_batch_status,
    trigger_aggregate_async,
)

file_upload_status_router = APIRouter(prefix="/api", tags=["文件上传状态"])


@file_upload_status_router.get("/upload-files/status", status_code=status.HTTP_200_OK)
async def upload_files_status_api(
    task_id: str = Query(..., description="上传任务ID"),
    current_user: User = Depends(get_current_user),
):
    safe_task_id = (task_id or "").strip()
    if not safe_task_id:
        raise HTTPException(status_code=400, detail="task_id 不能为空")

    try:
        data = get_mgf_only_batch_status(user_data_path=current_user.data_path, task_id=safe_task_id)
    except SiriusBatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # 如果所有 Celery 任务已完成但聚合尚未执行，异步触发聚合（不阻塞本次响应）
    progress = data.get("progress") or {}
    if progress.get("done") == progress.get("total") and not data.get("files"):
        await trigger_aggregate_async(current_user.data_path, safe_task_id)

    return success_response(message="上传任务状态", data=data)
