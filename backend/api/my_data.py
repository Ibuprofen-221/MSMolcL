import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from core.auth import get_current_user
from core.response import success_response
from models.user import User
from schemas.user_data import MyDataUpdateRequest
from util.file_utils import resolve_user_data_dir

my_data_router = APIRouter(tags=["我的数据"])


MY_DATA_FILENAME = "my_data.json"


def _my_data_path(user: User) -> Path:
    user_dir = resolve_user_data_dir(user.data_path)
    return user_dir / MY_DATA_FILENAME


def _write_json_atomically(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(target)


@my_data_router.get("/my-data", status_code=status.HTTP_200_OK)
async def get_my_data_api(current_user: User = Depends(get_current_user)):
    path = _my_data_path(current_user)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="用户数据文件不存在")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="读取用户数据失败") from None

    return success_response(message="读取成功", data=data)


@my_data_router.put("/my-data", status_code=status.HTTP_200_OK)
async def put_my_data_api(
    payload: MyDataUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    path = _my_data_path(current_user)

    try:
        _write_json_atomically(path, payload.data)
    except Exception:
        raise HTTPException(status_code=500, detail="写入用户数据失败") from None

    return success_response(message="写入成功", data=payload.data)
