from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from core.config import file_size_limit, task_storage_roots, user_data_root
from core.exceptions import FileSizeError

# 任务文件根目录（默认：普通检索目录）
BASE_FILE_DIR: Path = task_storage_roots["normal"]


def _get_storage_root(storage: str = "normal") -> Path:
    root = task_storage_roots.get(storage)
    if root is None:
        raise ValueError(f"不支持的任务空间: {storage}")
    return root


def generate_task_id() -> str:
    """生成全局唯一 Task ID。"""
    return uuid4().hex


def _safe_task_id(task_id: str) -> str:
    safe_task_id = (task_id or "").strip()
    if not safe_task_id:
        raise ValueError("task_id不能为空")
    if not all(c.isalnum() or c in {"-", "_"} for c in safe_task_id):
        raise ValueError("非法task_id")
    return safe_task_id


def _safe_username(username: str) -> str:
    value = (username or "").strip()
    if not value:
        raise ValueError("用户名不能为空")
    if not all(c.isalnum() or c in {"-", "_"} for c in value):
        raise ValueError("用户名仅支持字母数字下划线与中划线")
    return value


def to_user_data_relative_path(username: str) -> str:
    return _safe_username(username)


def resolve_user_data_dir(user_data_path: str) -> Path:
    safe_relative = (user_data_path or "").strip()
    if not safe_relative:
        raise ValueError("用户数据路径不能为空")

    root = user_data_root.resolve()
    target = (user_data_root / safe_relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("非法用户数据路径") from None

    return target


def init_user_data_layout(user_data_path: str) -> Path:
    user_dir = resolve_user_data_dir(user_data_path)
    user_dir.mkdir(parents=True, exist_ok=True)

    history_path = user_dir / "history_records.json"
    if not history_path.exists():
        history_path.write_text("[]", encoding="utf-8")

    my_data_path = user_dir / "my_data.json"
    if not my_data_path.exists():
        my_data_path.write_text("{}", encoding="utf-8")

    return user_dir


def get_user_task_dir(user_data_path: str, task_id: str, storage: str = "normal", create: bool = False) -> Path:
    _ = storage  # 当前 normal/advanced 统一映射到同一用户目录
    user_dir = resolve_user_data_dir(user_data_path)
    safe_task_id = _safe_task_id(task_id)

    task_dir = (user_dir / f"task_{safe_task_id}").resolve()
    try:
        task_dir.relative_to(user_dir)
    except ValueError:
        raise ValueError("非法task_id") from None

    if create:
        task_dir.mkdir(parents=True, exist_ok=True)

    return task_dir


def get_user_file_path(
    user_data_path: str,
    task_id: str,
    filename: str,
    storage: str = "normal",
    create: bool = False,
) -> Path:
    safe_filename = Path(filename or "").name
    if not safe_filename or safe_filename in {".", ".."}:
        raise ValueError("非法文件名")

    task_dir = get_user_task_dir(user_data_path, task_id, storage=storage, create=create)
    return task_dir / safe_filename


def get_user_file_path_readonly(
    user_data_path: str,
    task_id: str,
    filename: str,
    storage: str = "normal",
) -> Path:
    return get_user_file_path(user_data_path, task_id, filename, storage=storage, create=False)


def create_task_dir(task_id: str, storage: str = "normal") -> Path:
    """兼容旧函数：基于 Task ID 创建并返回任务目录路径。"""
    base_dir = _get_storage_root(storage)
    task_dir = base_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def get_file_path(task_id: str, filename: str, storage: str = "normal") -> Path:
    """兼容旧函数：获取 Task ID 下的文件完整路径。"""
    return create_task_dir(task_id, storage=storage) / filename


def get_file_path_readonly(task_id: str, filename: str, storage: str = "normal") -> Path:
    """兼容旧函数：只拼接文件路径，不创建目录。"""
    base_dir = _get_storage_root(storage)
    return base_dir / f"task_{task_id}" / filename


async def save_upload_file(upload_file: UploadFile, target_path: Path) -> None:
    """流式保存上传文件，并执行100MB大小限制校验。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    current_size = 0
    with open(target_path, "wb") as output_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break

            current_size += len(chunk)
            if current_size > file_size_limit:
                output_file.close()
                safe_remove_file(target_path)
                raise FileSizeError("文件大小超过100MB限制")

            output_file.write(chunk)


def get_file_extension(filename: str | None) -> str:
    """获取文件后缀（小写，不含点）。"""
    if not filename:
        return ""
    return Path(filename).suffix.lstrip(".").lower()


def safe_remove_file(file_path: Path) -> None:
    """安全删除文件（存在才删除）。"""
    if file_path.exists() and file_path.is_file():
        file_path.unlink()
