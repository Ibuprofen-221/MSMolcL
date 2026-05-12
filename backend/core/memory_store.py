import time
from threading import Lock

from core.config import custom_lib_cache_ttl_seconds

# 自定义候选库缓存（任务级 TTL，线程安全）
_custom_lib_lock = Lock()
_custom_lib_store: dict[str, tuple[float, list[str]]] = {}


def set_custom_lib_cache(task_id: str, smiles_list: list[str]) -> None:
    with _custom_lib_lock:
        key = f"task:{task_id}"
        expire_at = time.monotonic() + custom_lib_cache_ttl_seconds
        _custom_lib_store[key] = (expire_at, smiles_list)


def get_custom_lib_cache(task_id: str) -> list[str] | None:
    with _custom_lib_lock:
        key = f"task:{task_id}"
        entry = _custom_lib_store.get(key)
        if entry is None:
            return None
        expire_at, smiles = entry
        if time.monotonic() > expire_at:
            del _custom_lib_store[key]
            return None
        return smiles
