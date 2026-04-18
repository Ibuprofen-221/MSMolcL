import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from core.auth import get_current_user
from core.config import allowed_json_extensions, allowed_mgf_extensions, rate_limit_upload_rule
from core.exceptions import FileFormatError, FileMissingError, FileSizeError
from core.memory_store import update_processed_cache
from core.rate_limit import limiter
from core.response import success_response
from models.user import User
from services.file_preprocess import main as preprocess_main
from services.history_store import upsert_task_record
from util.file_utils import (
    generate_task_id,
    get_file_extension,
    get_user_file_path,
    safe_remove_file,
    save_upload_file,
)

file_upload_router = APIRouter(prefix="/api", tags=["文件上传"])


def _validate_upload_files(file_mgf: UploadFile | None, file_json: UploadFile | None) -> None:
    """执行必传与格式校验。"""
    if file_mgf is None or file_json is None:
        raise FileMissingError("必须同时上传mgf/txt文件和json文件")

    mgf_ext = get_file_extension(file_mgf.filename)
    json_ext = get_file_extension(file_json.filename)

    if mgf_ext not in allowed_mgf_extensions:
        raise FileFormatError("file_mgf仅支持mgf/txt格式")

    if json_ext not in allowed_json_extensions:
        raise FileFormatError("file_json仅支持json格式")


@file_upload_router.post("/upload-files", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_upload_rule)
async def upload_files(
    request: Request,
    file_mgf: UploadFile | None = File(default=None),
    file_json: UploadFile | None = File(default=None),
    current_user: User = Depends(get_current_user),
) -> dict:
    """上传并处理mgf/txt与json文件（按任务目录隔离输出）。"""
    task_id = generate_task_id()
    saved_mgf_path: Path | None = None
    saved_json_path: Path | None = None

    try:
        _validate_upload_files(file_mgf=file_mgf, file_json=file_json)

        mgf_ext = get_file_extension(file_mgf.filename)
        json_ext = get_file_extension(file_json.filename)
        unique_suffix = str(int(time.time() * 1000))

        saved_mgf_path = get_user_file_path(
            current_user.data_path,
            task_id,
            f"upload_mgf_{unique_suffix}.{mgf_ext}",
            create=True,
        )
        saved_json_path = get_user_file_path(
            current_user.data_path,
            task_id,
            f"upload_json_{unique_suffix}.{json_ext}",
            create=True,
        )

        await save_upload_file(upload_file=file_mgf, target_path=saved_mgf_path)
        await save_upload_file(upload_file=file_json, target_path=saved_json_path)

        task_output_dir = saved_mgf_path.parent
        statas_file = get_user_file_path(current_user.data_path, task_id, "statas.json")
        fragtrees_file = get_user_file_path(current_user.data_path, task_id, "valid_pairs_fragtrees.json")
        spectra_file = get_user_file_path(current_user.data_path, task_id, "valid_pairs_spectra.mgf")

        preprocess_main(str(saved_mgf_path), str(saved_json_path), output_base_dir=task_output_dir)

        if not spectra_file.exists() or not fragtrees_file.exists() or not statas_file.exists():
            raise HTTPException(status_code=500, detail="文件处理结果不完整")

        valid_pairs_spectra_content = spectra_file.read_text(encoding="utf-8")
        valid_pairs_fragtrees_content = json.loads(fragtrees_file.read_text(encoding="utf-8"))
        statas_data = json.loads(statas_file.read_text(encoding="utf-8"))

        update_processed_cache(
            spectra_content=valid_pairs_spectra_content,
            fragtrees_content=valid_pairs_fragtrees_content,
            statas=statas_data,
        )

        upsert_task_record(
            task_id=task_id,
            user_data_path=current_user.data_path,
            normal_status="pending",
            advanced_status="pending",
        )

        return success_response(
            data={
                "task_id": task_id,
                "statas": statas_data,
                "output_files": {
                    "valid_pairs_spectra": "valid_pairs_spectra.mgf",
                    "valid_pairs_fragtrees": "valid_pairs_fragtrees.json",
                    "statas": "statas.json",
                },
                "files": {
                    "statas_path": str(statas_file.resolve()),
                    "fragtrees_path": str(fragtrees_file.resolve()),
                    "spectra_path": str(spectra_file.resolve()),
                },
            },
            message="文件上传并处理成功",
        )
    except (FileMissingError, FileFormatError, FileSizeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="文件处理失败") from None
    finally:
        if file_mgf is not None:
            await file_mgf.close()
        if file_json is not None:
            await file_json.close()

        if saved_mgf_path is not None:
            safe_remove_file(saved_mgf_path)
        if saved_json_path is not None:
            safe_remove_file(saved_json_path)


@file_upload_router.post("/upload-files-advanced", status_code=status.HTTP_400_BAD_REQUEST)
async def upload_files_advanced() -> dict:
    """高级检索改为 task_id 模式，已下线文件上传入口。"""
    raise HTTPException(status_code=400, detail="高级检索不再支持上传文件，请在高级检索页面输入任务ID")
