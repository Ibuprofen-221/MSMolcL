import json
import shutil
from datetime import datetime
from pathlib import Path

from util.file_utils import get_user_task_dir, resolve_user_data_dir

ALLOWED_STATUS = {"pending", "success", "failed"}
DEFAULT_NOTE = "No notes"


def _normalize_note(note: str | None) -> str:
    text = (note or "").strip()
    return text if text else DEFAULT_NOTE


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _history_file_path(user_data_path: str) -> Path:
    user_dir = resolve_user_data_dir(user_data_path)
    return user_dir / "history_records.json"


def _ensure_parent(history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)


def _read_data(user_data_path: str) -> list[dict]:
    history_path = _history_file_path(user_data_path)
    _ensure_parent(history_path)
    if not history_path.exists():
        return []

    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        raw_records = data
    elif isinstance(data, dict):
        records = data.get("records")
        raw_records = records if isinstance(records, list) else []
    else:
        raw_records = []

    normalized_records: list[dict] = []
    for item in raw_records:
        if not isinstance(item, dict):
            continue

        task_id = str(item.get("task_id") or "").strip()
        if not task_id:
            continue

        normal_status = item.get("normal_status")
        if normal_status not in ALLOWED_STATUS:
            normal_status = "pending"

        advanced_status = item.get("advanced_status")
        if advanced_status not in ALLOWED_STATUS:
            advanced_status = "pending"

        create_time = item.get("create_time")
        if not isinstance(create_time, str) or not create_time.strip():
            create_time = _utc_now_iso()

        normalized_records.append(
            {
                "task_id": task_id,
                "create_time": create_time,
                "normal_status": normal_status,
                "advanced_status": advanced_status,
                "note": _normalize_note(item.get("note")),
            }
        )

    return normalized_records


def _atomic_write(user_data_path: str, records: list[dict]) -> None:
    history_path = _history_file_path(user_data_path)
    _ensure_parent(history_path)
    tmp_path = Path(f"{history_path}.tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(history_path)


def _resolve_task_dir(user_data_path: str, task_id: str) -> Path:
    return get_user_task_dir(user_data_path=user_data_path, task_id=task_id, create=False)


def _remove_task_dir_if_exists(user_data_path: str, task_id: str) -> bool:
    target_dir = _resolve_task_dir(user_data_path, task_id)
    if not target_dir.exists():
        return False
    if not target_dir.is_dir():
        raise ValueError("任务目录异常，删除失败")
    shutil.rmtree(target_dir)
    return True


def upsert_task_record(
    task_id: str,
    user_data_path: str,
    normal_status: str | None = None,
    advanced_status: str | None = None,
    note: str | None = None,
) -> dict:
    if not task_id or not task_id.strip():
        raise ValueError("task_id不能为空")

    if normal_status is not None and normal_status not in ALLOWED_STATUS:
        raise ValueError("normal_status 仅支持 pending/success/failed")
    if advanced_status is not None and advanced_status not in ALLOWED_STATUS:
        raise ValueError("advanced_status 仅支持 pending/success/failed")

    records = _read_data(user_data_path)

    target = None
    for item in records:
        if isinstance(item, dict) and item.get("task_id") == task_id:
            target = item
            break

    if target is None:
        target = {
            "task_id": task_id,
            "create_time": _utc_now_iso(),
            "normal_status": "pending",
            "advanced_status": "pending",
            "note": _normalize_note(note),
        }
        records.append(target)
    else:
        target["note"] = _normalize_note(target.get("note"))

    if normal_status is not None:
        target["normal_status"] = normal_status
    if advanced_status is not None:
        target["advanced_status"] = advanced_status
    if note is not None:
        target["note"] = _normalize_note(note)

    _atomic_write(user_data_path, records)
    return target


def delete_task_record(task_id: str, user_data_path: str) -> dict:
    safe_task_id = (task_id or "").strip()
    if not safe_task_id:
        raise ValueError("task_id不能为空")

    records = _read_data(user_data_path)

    record_removed = False
    filtered_records: list[dict] = []
    for item in records:
        if isinstance(item, dict) and item.get("task_id") == safe_task_id:
            record_removed = True
            continue
        filtered_records.append(item)

    dir_removed = _remove_task_dir_if_exists(user_data_path, safe_task_id)

    if record_removed:
        _atomic_write(user_data_path, filtered_records)

    if not record_removed and not dir_removed:
        raise ValueError("历史记录不存在")

    return {
        "task_id": safe_task_id,
        "record_deleted": record_removed,
        "task_dir_deleted": dir_removed,
    }


def list_user_records(user_data_path: str) -> list[dict]:
    records = _read_data(user_data_path)

    def _sort_key(item: dict) -> str:
        value = item.get("create_time")
        return value if isinstance(value, str) else ""

    safe_records = [item for item in records if isinstance(item, dict)]
    return sorted(safe_records, key=_sort_key, reverse=True)
