import json
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth import get_current_user
from core.config import batch_group_ttl_seconds, rate_limit_retrieve_rule, rate_limit_retrieve_status_rule
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
BATCH_GROUP_PREFIX = "batchgrp_"


class _TTLDict:
    """带 TTL 的字典，访问时惰性清理过期条目。"""

    def __init__(self, ttl_seconds: float = 7200):
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, dict]] = {}
        self._lock = Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            created_at, value = entry
            if time.monotonic() - created_at > self._ttl:
                del self._data[key]
                return None
            return value

    def __setitem__(self, key: str, value: dict):
        with self._lock:
            self._data[key] = (time.monotonic(), value)

    def __delitem__(self, key: str):
        with self._lock:
            self._data.pop(key, None)

    def cleanup(self):
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (ts, _) in self._data.items() if now - ts > self._ttl]
            for k in expired:
                del self._data[k]


BATCH_JOB_GROUPS = _TTLDict(ttl_seconds=batch_group_ttl_seconds)


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
    if status_value not in {"success", "failed", "partial_failed"}:
        return

    persisted_status = "failed" if status_value == "partial_failed" else status_value
    if stage == "normal":
        upsert_task_record(task_id=task_id, user_data_path=user_data_path, normal_status=persisted_status)
    elif stage == "advanced":
        upsert_task_record(task_id=task_id, user_data_path=user_data_path, advanced_status=persisted_status)


def _ensure_task_belongs_to_user(user_data_path: str, task_id: str) -> bool:
    try:
        task_dir = get_user_task_dir(user_data_path=user_data_path, task_id=task_id, create=False)
    except ValueError:
        return False
    return task_dir.exists() and task_dir.is_dir()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_batch_items(user_data_path: str, task_id: str) -> list[dict]:
    statas_path = get_user_file_path_readonly(user_data_path, task_id, NORMAL_STATAS_FILENAME, storage="normal")
    statas_data = _read_json(statas_path)
    items = statas_data.get("批次文件")
    if not isinstance(items, list):
        return []
    safe_items = [item for item in items if isinstance(item, dict)]
    return safe_items


def _build_normal_entries(user_data_path: str, task_id: str) -> tuple[list[dict], list[dict]]:
    batch_items = _load_batch_items(user_data_path, task_id)
    if batch_items:
        entries: list[dict] = []
        for item in batch_items:
            files = item.get("output_files") or {}
            statas_path = Path(str(files.get("statas") or ""))
            fragtrees_path = Path(str(files.get("fragtrees") or ""))
            spectra_path = Path(str(files.get("spectra") or ""))
            if statas_path.is_file() and fragtrees_path.is_file() and spectra_path.is_file():
                entries.append(
                    {
                        "pair_key": str(item.get("pair_key") or ""),
                        "statas_path": statas_path,
                        "fragtrees_path": fragtrees_path,
                        "spectra_path": spectra_path,
                    }
                )
        if entries:
            return entries, batch_items

    statas_path = get_user_file_path_readonly(user_data_path, task_id, NORMAL_STATAS_FILENAME, storage="normal")
    fragtrees_path = get_user_file_path_readonly(user_data_path, task_id, "valid_pairs_fragtrees.json", storage="normal")
    spectra_path = get_user_file_path_readonly(user_data_path, task_id, "valid_pairs_spectra.mgf", storage="normal")
    return (
        [
            {
                "pair_key": "",
                "statas_path": statas_path,
                "fragtrees_path": fragtrees_path,
                "spectra_path": spectra_path,
            }
        ],
        [],
    )


def _build_advanced_entries(user_data_path: str, task_id: str) -> tuple[list[dict], list[dict]]:
    batch_items = _load_batch_items(user_data_path, task_id)
    if batch_items:
        entries: list[dict] = []
        for item in batch_items:
            files = item.get("output_files") or {}
            normal_statas_path = Path(str(files.get("statas") or ""))
            fragtrees_path = Path(str(files.get("fragtrees") or ""))
            spectra_path = Path(str(files.get("spectra") or ""))
            if not (normal_statas_path.is_file() and fragtrees_path.is_file() and spectra_path.is_file()):
                continue

            advanced_statas_path = normal_statas_path.parent / ADVANCED_STATAS_FILENAME
            advanced_statas_path.write_text(normal_statas_path.read_text(encoding="utf-8"), encoding="utf-8")
            entries.append(
                {
                    "pair_key": str(item.get("pair_key") or ""),
                    "statas_path": advanced_statas_path,
                    "fragtrees_path": fragtrees_path,
                    "spectra_path": spectra_path,
                }
            )

        if entries:
            return entries, batch_items

    normal_statas_path = get_user_file_path_readonly(user_data_path, task_id, NORMAL_STATAS_FILENAME, storage="normal")
    advanced_statas_path = get_user_file_path_readonly(user_data_path, task_id, ADVANCED_STATAS_FILENAME, storage="normal")
    fragtrees_path = get_user_file_path_readonly(user_data_path, task_id, "valid_pairs_fragtrees.json", storage="normal")
    spectra_path = get_user_file_path_readonly(user_data_path, task_id, "valid_pairs_spectra.mgf", storage="normal")
    if normal_statas_path.is_file():
        advanced_statas_path.write_text(normal_statas_path.read_text(encoding="utf-8"), encoding="utf-8")

    return (
        [
            {
                "pair_key": "",
                "statas_path": advanced_statas_path,
                "fragtrees_path": fragtrees_path,
                "spectra_path": spectra_path,
            }
        ],
        [],
    )


def _validate_entry_files(entries: list[dict]) -> str | None:
    if not entries:
        return "任务文件缺失"
    for entry in entries:
        for file_path in [entry.get("statas_path"), entry.get("fragtrees_path"), entry.get("spectra_path")]:
            if not isinstance(file_path, Path) or not file_path.is_file():
                return f"任务文件缺失: {getattr(file_path, 'name', 'unknown')}"
    return None


def _register_batch_group(
    task_id: str,
    user_data_path: str,
    stage: str,
    job_ids: list[str],
    entries: list[dict],
    batch_items: list[dict],
) -> str:
    group_id = f"{BATCH_GROUP_PREFIX}{uuid4().hex}"
    BATCH_JOB_GROUPS[group_id] = {
        "group_id": group_id,
        "task_id": task_id,
        "user_data_path": user_data_path,
        "stage": stage,
        "job_ids": job_ids,
        "entries": entries,
        "batch_items": batch_items,
        "merged": False,
    }
    return group_id


def _aggregate_group_status(group: dict, runtime_getter) -> dict:
    child_jobs = []
    for child_id in group.get("job_ids", []):
        item = runtime_getter(child_id)
        if item is None:
            child_jobs.append({"job_id": child_id, "status": "failed", "error": "子任务不存在或已过期"})
        else:
            child_jobs.append(item)

    statuses = [str(item.get("status") or "") for item in child_jobs]
    total_count = len(child_jobs)
    failed_count = sum(1 for status in statuses if status == "failed")
    success_count = sum(1 for status in statuses if status == "success")

    if any(status in {"pending", "running"} for status in statuses):
        overall = "running"
    elif failed_count > 0 and success_count > 0:
        overall = "partial_failed"
    elif failed_count > 0:
        overall = "failed"
    elif child_jobs:
        overall = "success"
    else:
        overall = "failed"

    return {
        "job_id": group.get("group_id"),
        "task_id": group.get("task_id"),
        "status": overall,
        "is_batch": True,
        "children": child_jobs,
        "failed_count": failed_count,
        "success_count": success_count,
        "total_count": total_count,
    }


def _merge_batch_outputs(group: dict) -> None:
    if group.get("merged"):
        return

    task_id = str(group.get("task_id") or "")
    user_data_path = str(group.get("user_data_path") or "")
    stage = str(group.get("stage") or "normal")
    entries = group.get("entries") or []

    merged_root_info: list[dict] = []
    merged_valid_pairs: list[str] = []
    merged_fragtrees: dict = {}
    merged_spectra_text_parts: list[str] = []

    for entry in entries:
        statas_path = Path(entry.get("statas_path"))
        fragtrees_path = Path(entry.get("fragtrees_path"))
        spectra_path = Path(entry.get("spectra_path"))

        if not (statas_path.is_file() and fragtrees_path.is_file() and spectra_path.is_file()):
            continue

        statas_data = _read_json(statas_path)
        root_info = statas_data.get("碎裂树文件统计", {}).get("有效碎裂树根节点信息", [])
        valid_pairs = statas_data.get("最终有效对", [])

        if isinstance(root_info, list):
            merged_root_info.extend([item for item in root_info if isinstance(item, dict)])
        if isinstance(valid_pairs, list):
            merged_valid_pairs.extend([item for item in valid_pairs if isinstance(item, str)])

        fragtrees_data = _read_json(fragtrees_path)
        merged_fragtrees.update(fragtrees_data)

        spectra_text = spectra_path.read_text(encoding="utf-8").strip()
        if spectra_text:
            merged_spectra_text_parts.append(spectra_text)

    normal_statas_path = get_user_file_path_readonly(user_data_path, task_id, NORMAL_STATAS_FILENAME, storage="normal")
    normal_statas_data = _read_json(normal_statas_path)
    batch_items = normal_statas_data.get("批次文件") if isinstance(normal_statas_data.get("批次文件"), list) else []
    batch_stats = normal_statas_data.get("批次文件统计") if isinstance(normal_statas_data.get("批次文件统计"), dict) else {}

    merged_statas = {
        "谱图文件统计": {
            "原始文件中的谱图数量": len(merged_root_info),
            "原始谱图title": [item.get("title", "") for item in merged_root_info],
            "解析的原始块数量": len(merged_root_info),
        },
        "碎裂树文件统计": {
            "有效谱图title": [item.get("title", "") for item in merged_root_info],
            "有效碎裂树根节点信息": merged_root_info,
        },
        "最终有效对": merged_valid_pairs,
        "最终有效对数量": len(merged_valid_pairs),
        "批次文件": batch_items,
        "批次文件统计": batch_stats,
    }

    target_statas_name = NORMAL_STATAS_FILENAME if stage == "normal" else ADVANCED_STATAS_FILENAME
    target_statas_path = get_user_file_path_readonly(user_data_path, task_id, target_statas_name, storage="normal")
    target_statas_path.write_text(json.dumps(merged_statas, ensure_ascii=False, indent=2), encoding="utf-8")

    if stage == "normal":
        target_fragtrees_path = get_user_file_path_readonly(user_data_path, task_id, "valid_pairs_fragtrees.json", storage="normal")
        target_spectra_path = get_user_file_path_readonly(user_data_path, task_id, "valid_pairs_spectra.mgf", storage="normal")
        target_fragtrees_path.write_text(json.dumps(merged_fragtrees, ensure_ascii=False, indent=2), encoding="utf-8")
        target_spectra_path.write_text("\n\n".join(merged_spectra_text_parts) + ("\n" if merged_spectra_text_parts else ""), encoding="utf-8")

    group["merged"] = True


@retrieve_router.post("/retrieve", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_rule)
async def retrieve_api(
    request: Request,
    payload: RetrievePayload,
    current_user: User = Depends(get_current_user),
):
    """异步检索提交接口：批次任务按文件拆分子job，兼容单job返回。"""
    _ = request
    entries, batch_items = _build_normal_entries(current_user.data_path, payload.task_id)

    if not entries or not entries[0]["statas_path"].parent.exists():
        return JSONResponse(status_code=400, content=error_response(message="任务ID不存在", code=400))

    missing_msg = _validate_entry_files(entries)
    if missing_msg:
        return JSONResponse(status_code=400, content=error_response(message=missing_msg, code=400))

    try:
        if len(entries) == 1:
            entry = entries[0]
            job_payload = build_retrieve_job_payload(
                search_type=payload.search_type,
                ppm_range=payload.ppm_range,
                custom_lib_file_name=payload.custom_lib_file_name,
                ion_mode=payload.ion_mode,
                statas_path=str(entry["statas_path"]),
                fragtrees_path=str(entry["fragtrees_path"]),
                spectra_path=str(entry["spectra_path"]),
                databases=payload.databases,
                task_id=payload.task_id,
            )
            job_info = retrieve_runtime.submit(job_payload=job_payload, task_id=payload.task_id)
            upsert_task_record(task_id=payload.task_id, user_data_path=current_user.data_path, normal_status="pending")
            return success_response(message="检索任务已提交", data=job_info)

        child_job_ids: list[str] = []
        for entry in entries:
            job_payload = build_retrieve_job_payload(
                search_type=payload.search_type,
                ppm_range=payload.ppm_range,
                custom_lib_file_name=payload.custom_lib_file_name,
                ion_mode=payload.ion_mode,
                statas_path=str(entry["statas_path"]),
                fragtrees_path=str(entry["fragtrees_path"]),
                spectra_path=str(entry["spectra_path"]),
                databases=payload.databases,
                task_id=payload.task_id,
            )
            child_job = retrieve_runtime.submit(job_payload=job_payload, task_id=payload.task_id)
            child_job_ids.append(str(child_job.get("job_id") or ""))

        group_id = _register_batch_group(
            task_id=payload.task_id,
            user_data_path=current_user.data_path,
            stage="normal",
            job_ids=child_job_ids,
            entries=entries,
            batch_items=batch_items,
        )

        upsert_task_record(task_id=payload.task_id, user_data_path=current_user.data_path, normal_status="pending")
        return success_response(
            message="检索批次任务已提交",
            data={
                "job_id": group_id,
                "task_id": payload.task_id,
                "status": "pending",
                "is_batch": True,
                "job_ids": child_job_ids,
            },
        )
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content=error_response(message=str(exc), code=400))
    except Exception as exc:
        return JSONResponse(status_code=500, content=error_response(message=f"检索任务提交失败: {exc}", code=500))


@retrieve_router.get("/retrieve/status", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_status_rule)
async def retrieve_status_api(
    request: Request,
    job_id: str = Query(..., description="检索任务ID"),
    current_user: User = Depends(get_current_user),
):
    """查询检索任务状态与结果。"""
    _ = request

    group = BATCH_JOB_GROUPS.get(job_id)
    if group is not None and str(group.get("stage")) == "normal":
        task_id = str(group.get("task_id") or "")
        if not _ensure_task_belongs_to_user(current_user.data_path, task_id):
            return JSONResponse(status_code=404, content=error_response(message="任务不存在或无权限", code=404))

        data = _aggregate_group_status(group, retrieve_runtime.get_job)
        if data["status"] in {"success", "partial_failed", "failed"}:
            _merge_batch_outputs(group)
            _sync_history_by_job(task_id=task_id, status_value=data["status"], stage="normal", user_data_path=current_user.data_path)
        return success_response(message="检索任务状态", data=data)

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
    """高级检索提交接口：批次任务按文件拆分子job，兼容单job返回。"""
    _ = request
    entries, batch_items = _build_advanced_entries(current_user.data_path, payload.task_id)

    if not entries or not entries[0]["statas_path"].parent.exists():
        return JSONResponse(status_code=400, content=error_response(message="任务ID不存在", code=400))

    missing_msg = _validate_entry_files(entries)
    if missing_msg:
        return JSONResponse(status_code=400, content=error_response(message=missing_msg, code=400))

    try:
        if len(entries) == 1:
            entry = entries[0]
            job_payload = build_retrieve_advanced_job_payload(
                ion_mode=payload.ion_mode,
                statas_path=str(entry["statas_path"]),
                fragtrees_path=str(entry["fragtrees_path"]),
                spectra_path=str(entry["spectra_path"]),
            )
            job_info = retrieve_advanced_runtime.submit(job_payload=job_payload, task_id=payload.task_id)
            upsert_task_record(task_id=payload.task_id, user_data_path=current_user.data_path, advanced_status="pending")
            return success_response(message="高级检索任务已提交", data=job_info)

        child_job_ids: list[str] = []
        for entry in entries:
            job_payload = build_retrieve_advanced_job_payload(
                ion_mode=payload.ion_mode,
                statas_path=str(entry["statas_path"]),
                fragtrees_path=str(entry["fragtrees_path"]),
                spectra_path=str(entry["spectra_path"]),
            )
            child_job = retrieve_advanced_runtime.submit(job_payload=job_payload, task_id=payload.task_id)
            child_job_ids.append(str(child_job.get("job_id") or ""))

        group_id = _register_batch_group(
            task_id=payload.task_id,
            user_data_path=current_user.data_path,
            stage="advanced",
            job_ids=child_job_ids,
            entries=entries,
            batch_items=batch_items,
        )

        upsert_task_record(task_id=payload.task_id, user_data_path=current_user.data_path, advanced_status="pending")
        return success_response(
            message="高级检索批次任务已提交",
            data={
                "job_id": group_id,
                "task_id": payload.task_id,
                "status": "pending",
                "is_batch": True,
                "job_ids": child_job_ids,
            },
        )
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content=error_response(message=str(exc), code=400))
    except Exception as exc:
        return JSONResponse(status_code=500, content=error_response(message=f"高级检索任务提交失败: {exc}", code=500))


@retrieve_router.get("/retrieve-advanced/status", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_retrieve_status_rule)
async def retrieve_advanced_status_api(
    request: Request,
    job_id: str = Query(..., description="高级检索任务ID"),
    current_user: User = Depends(get_current_user),
):
    """查询高级检索任务状态与结果。"""
    _ = request

    group = BATCH_JOB_GROUPS.get(job_id)
    if group is not None and str(group.get("stage")) == "advanced":
        task_id = str(group.get("task_id") or "")
        if not _ensure_task_belongs_to_user(current_user.data_path, task_id):
            return JSONResponse(status_code=404, content=error_response(message="任务不存在或无权限", code=404))

        data = _aggregate_group_status(group, retrieve_advanced_runtime.get_job)
        if data["status"] in {"success", "partial_failed", "failed"}:
            _merge_batch_outputs(group)
            _sync_history_by_job(task_id=task_id, status_value=data["status"], stage="advanced", user_data_path=current_user.data_path)
        return success_response(message="高级检索任务状态", data=data)

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
