from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from core.auth import get_current_user
from core.response import success_response
from models.user import User
from services.history_store import delete_task_record, list_user_records, upsert_task_record

history_router = APIRouter(prefix="/api", tags=["历史记录"])


class HistoryUpdatePayload(BaseModel):
    task_id: str
    normal_status: str | None = None
    advanced_status: str | None = None
    note: str | None = None


@history_router.get("/history", status_code=status.HTTP_200_OK)
async def history_list_api(current_user: User = Depends(get_current_user)):
    records = list_user_records(user_data_path=current_user.data_path)
    return success_response(message="历史记录读取成功", data={"username": current_user.username, "records": records})


@history_router.post("/history/update", status_code=status.HTTP_200_OK)
async def history_update_api(
    payload: HistoryUpdatePayload,
    current_user: User = Depends(get_current_user),
):
    try:
        record = upsert_task_record(
            task_id=payload.task_id,
            user_data_path=current_user.data_path,
            normal_status=payload.normal_status,
            advanced_status=payload.advanced_status,
            note=payload.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return success_response(message="历史记录更新成功", data=record)


@history_router.delete("/history/{task_id}", status_code=status.HTTP_200_OK)
async def history_delete_api(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    try:
        deleted = delete_task_record(task_id=task_id, user_data_path=current_user.data_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    return success_response(message="历史记录删除成功", data=deleted)
