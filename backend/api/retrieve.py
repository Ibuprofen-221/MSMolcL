from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth import get_current_user
from core.config import rate_limit_retrieve_rule, rate_limit_retrieve_status_rule
from core.rate_limit import limiter
from core.response import error_response, success_response
from models.user import User
from services.history_store import upsert_task_record
from services.retrieve_advanced_runtime import retrieve_advanced_runtime
from services.retrieve_advanced_service import build_retrieve_advanced_job_payload
from services.retrieve_runtime import retrieve_runtime
from services.retrieve_service import build_retrieve_job_payload
from util.file_utils import get_user_file_path_readonly, get_user_task_dir

retrieve_router = APIRouter(prefix="/api", tags=["检索"])

NORMAL_STATAS_FILENAME = "statas.json"
ADVANCED_STATAS_FILENAME = "statas_advanced.json"


class RetrievePayload(BaseModel):
    task_id: str
    search_type: str
    ion_mode: str = "pos"
    ppm_range: float | None = None
    custom_lib_file_name: str | None = None
    databases: list[str] | None = None


class RetrieveAdvancedPayload(BaseModel):
    task_id: str
    ion_mode: str = "pos"


def _sync_history_by_job(task_id: str, status_value: str, stage: str, user_data_path: str) -> None:
    if status_value not in {"success", "failed"}:
        return
    if stage == "normal":
        upsert_task_record(task_id=task_id, user_data_path=user_data_path, normal_status=status_value)
    elif stage == "advanced":
        upsert_task_record(task_id=task_id, user_data_path=user_data_path, advanced_status=status_value)


def _ensure_task_belongs_to_user(user_data_path: str, task_id: str) -> bool:
    try:
        task_dir = get_user_task_dir(user_data_path=user_data_path, task_id=task_id, create=False)
    except ValueError:
        return False
    return task_dir.exists() and task_dir.is_dir()


@retrieve_router.post("/retrieve", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_rule)
async def retrieve_api(
    request: Request,
    payload: RetrievePayload,
    current_user: User = Depends(get_current_user),
):
    """异步检索提交接口：返回 job_id，实际检索在后台进程执行。"""
    statas_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, NORMAL_STATAS_FILENAME, storage="normal")
    fragtrees_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, "valid_pairs_fragtrees.json", storage="normal")
    spectra_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, "valid_pairs_spectra.mgf", storage="normal")

    if not statas_path.parent.exists():
        return JSONResponse(status_code=400, content=error_response(message="任务ID不存在", code=400))

    for file_path in [statas_path, fragtrees_path, spectra_path]:
        if not file_path.is_file():
            return JSONResponse(
                status_code=400,
                content=error_response(message=f"任务文件缺失: {file_path.name}", code=400),
            )

    try:
        job_payload = build_retrieve_job_payload(
            search_type=payload.search_type,
            ppm_range=payload.ppm_range,
            custom_lib_file_name=payload.custom_lib_file_name,
            ion_mode=payload.ion_mode,
            statas_path=str(statas_path),
            fragtrees_path=str(fragtrees_path),
            spectra_path=str(spectra_path),
            databases=payload.databases,
        )
        job_info = retrieve_runtime.submit(job_payload=job_payload, task_id=payload.task_id)
        upsert_task_record(task_id=payload.task_id, user_data_path=current_user.data_path, normal_status="pending")
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content=error_response(message=str(exc), code=400))
    except Exception as exc:
        return JSONResponse(status_code=500, content=error_response(message=f"检索任务提交失败: {exc}", code=500))

    return success_response(message="检索任务已提交", data=job_info)


@retrieve_router.get("/retrieve/status", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_status_rule)
async def retrieve_status_api(
    request: Request,
    job_id: str = Query(..., description="检索任务ID"),
    current_user: User = Depends(get_current_user),
):
    """查询检索任务状态与结果。"""
    job = retrieve_runtime.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content=error_response(message="任务不存在或已过期", code=404))

    task_id = str(job.get("task_id") or "")
    if not _ensure_task_belongs_to_user(current_user.data_path, task_id):
        return JSONResponse(status_code=404, content=error_response(message="任务不存在或无权限", code=404))

    _sync_history_by_job(
        task_id=task_id,
        status_value=job.get("status", ""),
        stage="normal",
        user_data_path=current_user.data_path,
    )
    return success_response(message="检索任务状态", data=job)


@retrieve_router.post("/retrieve-advanced", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_rule)
async def retrieve_advanced_api(
    request: Request,
    payload: RetrieveAdvancedPayload,
    current_user: User = Depends(get_current_user),
):
    """高级检索提交接口：基于已有 task_id 的普通检索结果执行高级检索。"""
    normal_statas_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, NORMAL_STATAS_FILENAME, storage="normal")
    advanced_statas_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, ADVANCED_STATAS_FILENAME, storage="normal")
    fragtrees_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, "valid_pairs_fragtrees.json", storage="normal")
    spectra_path = get_user_file_path_readonly(current_user.data_path, payload.task_id, "valid_pairs_spectra.mgf", storage="normal")

    if not normal_statas_path.parent.exists():
        return JSONResponse(status_code=400, content=error_response(message="任务ID不存在", code=400))

    for file_path in [normal_statas_path, fragtrees_path, spectra_path]:
        if not file_path.is_file():
            return JSONResponse(
                status_code=400,
                content=error_response(message=f"任务文件缺失: {file_path.name}", code=400),
            )

    try:
        advanced_statas_path.write_text(normal_statas_path.read_text(encoding="utf-8"), encoding="utf-8")
        job_payload = build_retrieve_advanced_job_payload(
            ion_mode=payload.ion_mode,
            statas_path=str(advanced_statas_path),
            fragtrees_path=str(fragtrees_path),
            spectra_path=str(spectra_path),
        )
        job_info = retrieve_advanced_runtime.submit(job_payload=job_payload, task_id=payload.task_id)
        upsert_task_record(task_id=payload.task_id, user_data_path=current_user.data_path, advanced_status="pending")
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content=error_response(message=str(exc), code=400))
    except Exception as exc:
        return JSONResponse(status_code=500, content=error_response(message=f"高级检索任务提交失败: {exc}", code=500))

    return success_response(message="高级检索任务已提交", data=job_info)


@retrieve_router.get("/retrieve-advanced/status", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_status_rule)
async def retrieve_advanced_status_api(
    request: Request,
    job_id: str = Query(..., description="高级检索任务ID"),
    current_user: User = Depends(get_current_user),
):
    """查询高级检索任务状态与结果。"""
    job = retrieve_advanced_runtime.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content=error_response(message="任务不存在或已过期", code=404))

    task_id = str(job.get("task_id") or "")
    if not _ensure_task_belongs_to_user(current_user.data_path, task_id):
        return JSONResponse(status_code=404, content=error_response(message="任务不存在或无权限", code=404))

    _sync_history_by_job(
        task_id=task_id,
        status_value=job.get("status", ""),
        stage="advanced",
        user_data_path=current_user.data_path,
    )
    return success_response(message="高级检索任务状态", data=job)
