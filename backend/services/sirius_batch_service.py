import asyncio
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import (
    sirius_celery_broker_url,
    sirius_celery_result_backend,
    sirius_celery_task_name,
    sirius_fast_threshold,
    sirius_queue_fast,
    sirius_queue_slow,
)

from services.file_preprocess import main as preprocess_main
from services.history_store import upsert_task_record
from util.file_utils import get_user_file_path, get_user_task_dir

try:
    from celery import Celery
    from celery.result import AsyncResult
except Exception:  # pragma: no cover - 运行时依赖保护
    Celery = None
    AsyncResult = None

BATCH_META_FILENAME = "sirius_batch_meta.json"


class SiriusBatchError(ValueError):
    pass


@lru_cache(maxsize=1)
def _get_celery_app():
    if Celery is None:
        raise SiriusBatchError("后端未安装 celery 依赖，无法使用 mgf-only 异步模式")
    return Celery("backend_sirius_client", broker=sirius_celery_broker_url, backend=sirius_celery_result_backend)


def _count_spectra(mgf_path: Path) -> int:
    count = 0
    with mgf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().upper() == "BEGIN IONS":
                count += 1
    return count


def _route_queue(mgf_path: Path) -> str:
    return sirius_queue_fast if _count_spectra(mgf_path) <= sirius_fast_threshold else sirius_queue_slow


def _meta_path(task_dir: Path) -> Path:
    return task_dir / BATCH_META_FILENAME


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 异步聚合任务追踪
_aggregate_tasks: dict[str, asyncio.Task] = {}


async def trigger_aggregate_async(user_data_path: str, task_id: str) -> None:
    """异步触发聚合（不阻塞调用方）。"""
    task_key = f"{user_data_path}:{task_id}"
    if task_key in _aggregate_tasks:
        return

    async def _do_aggregate():
        try:
            task_dir = get_user_task_dir(user_data_path=user_data_path, task_id=task_id, create=False)
            meta = _read_meta(task_dir)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                _run_aggregate_if_needed,
                user_data_path, task_id, task_dir, meta,
            )
        finally:
            _aggregate_tasks.pop(task_key, None)

    _aggregate_tasks[task_key] = asyncio.create_task(_do_aggregate())


def _read_meta(task_dir: Path) -> dict:
    return _read_json(_meta_path(task_dir))


def _write_meta(task_dir: Path, data: dict) -> None:
    _write_json(_meta_path(task_dir), data)


def _resolve_generated_json(item: dict, result_data: dict | None = None) -> str:
    if isinstance(result_data, dict):
        output_file = str(result_data.get("output_file") or "").strip()
        if output_file and Path(output_file).is_file():
            return str(Path(output_file).resolve())

    pair_dir = Path(str(item.get("pair_dir") or ""))
    biz_task_id = str(item.get("biz_task_id") or "")
    if pair_dir.is_dir() and biz_task_id:
        matched = sorted(pair_dir.glob(f"*_{biz_task_id}_frag_trees.json"))
        if matched:
            return str(matched[0].resolve())
    return ""


def _calc_overall(items: list[dict]) -> tuple[str, dict]:
    total = len(items)
    success = sum(1 for x in items if str(x.get("status")) == "success")
    failed = sum(1 for x in items if str(x.get("status")) == "failed")
    running = sum(1 for x in items if str(x.get("status")) == "running")
    pending = total - success - failed - running
    done = success + failed

    if done < total:
        overall = "running" if running > 0 else "pending"
    else:
        if success == total:
            overall = "success"
        elif success > 0:
            overall = "partial_failed"
        else:
            overall = "failed"

    return overall, {
        "total": total,
        "done": done,
        "success": success,
        "failed": failed,
        "running": running,
        "pending": pending,
    }


def submit_mgf_only_batch(user_data_path: str, task_id: str, mgf_entries: list[dict]) -> dict:
    if not mgf_entries:
        raise SiriusBatchError("mgf-only 模式未收到可处理文件")

    task_dir = get_user_task_dir(user_data_path=user_data_path, task_id=task_id, create=True)
    app = _get_celery_app()

    items: list[dict] = []
    for entry in mgf_entries:
        pair_key = str(entry.get("pair_key") or "").strip()
        mgf_path = Path(str(entry.get("mgf_path") or "")).resolve()
        pair_dir = Path(str(entry.get("pair_dir") or "")).resolve()
        source_mgf_name = str(entry.get("source_mgf_name") or mgf_path.name)
        if not pair_key or not mgf_path.is_file() or not pair_dir.is_dir():
            raise SiriusBatchError(f"批次项不合法: {entry}")

        queue_name = _route_queue(mgf_path)
        biz_task_id = str(entry.get("biz_task_id") or f"sirius_{task_id}_{pair_key}")
        async_result = app.send_task(
            sirius_celery_task_name,
            args=[biz_task_id, str(mgf_path), str(pair_dir)],
            queue=queue_name,
        )

        items.append(
            {
                "pair_key": pair_key,
                "source_mgf_name": source_mgf_name,
                "pair_dir": str(pair_dir),
                "mgf_path": str(mgf_path),
                "queue": queue_name,
                "biz_task_id": biz_task_id,
                "celery_task_id": str(async_result.id),
                "status": "pending",
                "error": "",
                "generated_json_path": "",
            }
        )

    meta = {
        "task_id": task_id,
        "mode": "mgf_only_async",
        "aggregate_done": False,
        "aggregate_status": "pending",
        "aggregate_error": "",
        "items": items,
        "final_files": {},
        "batch_summary": {},
    }
    _write_meta(task_dir, meta)

    return {
        "task_id": task_id,
        "status": "processing",
        "upload_mode": "mgf_only_async",
        "progress": {
            "total": len(items),
            "done": 0,
            "success": 0,
            "failed": 0,
            "running": 0,
            "pending": len(items),
        },
    }


def _run_aggregate_if_needed(user_data_path: str, task_id: str, task_dir: Path, meta: dict) -> None:
    if bool(meta.get("aggregate_done")):
        return

    items = [x for x in meta.get("items", []) if isinstance(x, dict)]
    _, progress = _calc_overall(items)
    if progress["done"] < progress["total"]:
        return

    merged_root_info: list[dict] = []
    merged_valid_pairs: list[str] = []
    merged_fragtrees: dict = {}
    merged_spectra_text_parts: list[str] = []
    batch_items: list[dict] = []

    for item in items:
        if str(item.get("status")) != "success":
            continue
        try:
            pair_key = str(item.get("pair_key") or "")
            pair_dir = Path(str(item.get("pair_dir") or ""))
            mgf_path = Path(str(item.get("mgf_path") or ""))
            json_path = Path(str(item.get("generated_json_path") or ""))
            if not pair_key or not pair_dir.is_dir() or not mgf_path.is_file() or not json_path.is_file():
                raise SiriusBatchError(f"配对数据不完整: {pair_key}")

            preprocess_main(str(mgf_path), str(json_path), output_base_dir=pair_dir)

            pair_statas_path = pair_dir / "statas.json"
            pair_fragtrees_path = pair_dir / "valid_pairs_fragtrees.json"
            pair_spectra_path = pair_dir / "valid_pairs_spectra.mgf"
            if not pair_statas_path.is_file() or not pair_fragtrees_path.is_file() or not pair_spectra_path.is_file():
                raise SiriusBatchError(f"配对预处理结果不完整: {pair_key}")

            pair_statas = _read_json(pair_statas_path)
            pair_fragtrees = _read_json(pair_fragtrees_path)
            pair_spectra_text = pair_spectra_path.read_text(encoding="utf-8")

            pair_root_info = pair_statas.get("碎裂树文件统计", {}).get("有效碎裂树根节点信息", [])
            pair_valid_pairs = pair_statas.get("最终有效对", [])
            if isinstance(pair_root_info, list):
                merged_root_info.extend([x for x in pair_root_info if isinstance(x, dict)])
            if isinstance(pair_valid_pairs, list):
                merged_valid_pairs.extend([x for x in pair_valid_pairs if isinstance(x, str)])
            if isinstance(pair_fragtrees, dict):
                merged_fragtrees.update(pair_fragtrees)
            if pair_spectra_text.strip():
                merged_spectra_text_parts.append(pair_spectra_text.strip())

            batch_items.append(
                {
                    "pair_key": pair_key,
                    "pair_dir": pair_dir.name,
                    "source_files": {
                        "mgf": str(item.get("source_mgf_name") or mgf_path.name),
                        "json": Path(json_path).name,
                    },
                    "output_files": {
                        "statas": str(pair_statas_path.resolve()),
                        "fragtrees": str(pair_fragtrees_path.resolve()),
                        "spectra": str(pair_spectra_path.resolve()),
                    },
                    "valid_pairs_count": len([x for x in pair_valid_pairs if isinstance(x, str)]),
                }
            )
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)

    success_items = [x for x in items if str(x.get("status")) == "success"]
    failed_items = [x for x in items if str(x.get("status")) == "failed"]

    if not success_items:
        meta["aggregate_done"] = True
        meta["aggregate_status"] = "failed"
        meta["aggregate_error"] = "所有 mgf 生成或预处理均失败"
        meta["batch_summary"] = {
            "paired_count": 0,
            "unmatched_mgf": [str(x.get("pair_key") or "") for x in failed_items],
            "unmatched_json": [],
        }
        _write_meta(task_dir, meta)
        return

    statas_file = get_user_file_path(user_data_path, task_id, "statas.json", create=True)
    fragtrees_file = get_user_file_path(user_data_path, task_id, "valid_pairs_fragtrees.json", create=True)
    spectra_file = get_user_file_path(user_data_path, task_id, "valid_pairs_spectra.mgf", create=True)

    merged_statas = {
        "碎裂树文件统计": {
            "有效碎裂树根节点信息": merged_root_info,
        },
        "批次文件统计": {
            "mgf上传数": len(items),
            "json上传数": len(success_items),
            "配对成功数": len(success_items),
            "未配对mgf": [str(x.get("pair_key") or "") for x in failed_items],
            "未配对json": [],
        },
        "批次文件": batch_items,
    }

    statas_file.write_text(json.dumps(merged_statas, ensure_ascii=False, indent=2), encoding="utf-8")
    fragtrees_file.write_text(json.dumps(merged_fragtrees, ensure_ascii=False, indent=2), encoding="utf-8")
    spectra_file.write_text("\n\n".join(merged_spectra_text_parts) + ("\n" if merged_spectra_text_parts else ""), encoding="utf-8")

    upsert_task_record(
        task_id=task_id,
        user_data_path=user_data_path,
        normal_status="pending",
        advanced_status="pending",
    )

    meta["aggregate_done"] = True
    meta["aggregate_status"] = "success" if not failed_items else "partial_failed"
    meta["aggregate_error"] = ""
    meta["final_files"] = {
        "statas_path": str(statas_file.resolve()),
        "fragtrees_path": str(fragtrees_file.resolve()),
        "spectra_path": str(spectra_file.resolve()),
    }
    meta["batch_summary"] = {
        "paired_count": len(success_items),
        "unmatched_mgf": [str(x.get("pair_key") or "") for x in failed_items],
        "unmatched_json": [],
    }
    _write_meta(task_dir, meta)


def get_mgf_only_batch_status(user_data_path: str, task_id: str) -> dict:
    task_dir = get_user_task_dir(user_data_path=user_data_path, task_id=task_id, create=False)
    if not task_dir.exists() or not task_dir.is_dir():
        raise SiriusBatchError("任务不存在")

    meta = _read_meta(task_dir)
    if str(meta.get("mode") or "") != "mgf_only_async":
        raise SiriusBatchError("当前任务不是 mgf-only 异步任务")

    app = _get_celery_app()
    items = [x for x in meta.get("items", []) if isinstance(x, dict)]
    for item in items:
        current_status = str(item.get("status") or "pending")
        if current_status in {"success", "failed"}:
            continue

        celery_task_id = str(item.get("celery_task_id") or "").strip()
        if not celery_task_id:
            item["status"] = "failed"
            item["error"] = "celery_task_id 缺失"
            continue

        if AsyncResult is None:
            item["status"] = "failed"
            item["error"] = "celery AsyncResult 不可用"
            continue

        async_result = AsyncResult(celery_task_id, app=app)
        state = str(async_result.state or "PENDING").upper()
        if state in {"PENDING", "RECEIVED", "RETRY"}:
            item["status"] = "pending"
            continue
        if state in {"STARTED"}:
            item["status"] = "running"
            continue

        if not async_result.ready():
            item["status"] = "running"
            continue

        result_data = async_result.result if isinstance(async_result.result, dict) else {}
        worker_status = str(result_data.get("status") or "").upper()
        if worker_status == "SUCCESS" or async_result.successful():
            item["status"] = "success"
            item["error"] = ""
            item["generated_json_path"] = _resolve_generated_json(item, result_data)
            if not item["generated_json_path"]:
                item["status"] = "failed"
                item["error"] = "任务成功但未找到生成的json文件"
        else:
            item["status"] = "failed"
            item["error"] = str(result_data.get("error") or f"worker状态异常: {worker_status or state}")

    meta["items"] = items
    _write_meta(task_dir, meta)

    # 聚合改由 trigger_aggregate_async 异步触发，不再在此处阻塞
    meta = _read_meta(task_dir)
    items = [x for x in meta.get("items", []) if isinstance(x, dict)]

    overall, progress = _calc_overall(items)
    if progress["done"] == progress["total"] and not bool(meta.get("aggregate_done")):
        overall = "running"  # 等待异步聚合完成
    elif bool(meta.get("aggregate_done")):
        aggregate_status = str(meta.get("aggregate_status") or "")
        if aggregate_status in {"success", "partial_failed", "failed"}:
            overall = aggregate_status

    response_data: dict[str, Any] = {
        "task_id": task_id,
        "status": overall,
        "upload_mode": "mgf_only_async",
        "progress": progress,
        "files": meta.get("final_files") if isinstance(meta.get("final_files"), dict) else {},
        "batch_summary": meta.get("batch_summary") if isinstance(meta.get("batch_summary"), dict) else {},
        "errors": [
            {
                "pair_key": str(item.get("pair_key") or ""),
                "error": str(item.get("error") or ""),
            }
            for item in items
            if str(item.get("status")) == "failed" and str(item.get("error") or "").strip()
        ],
    }

    files = response_data.get("files") or {}
    statas_path = Path(str(files.get("statas_path") or "")) if isinstance(files, dict) else Path("")
    if statas_path.is_file():
        response_data["statas"] = _read_json(statas_path)

    return response_data
