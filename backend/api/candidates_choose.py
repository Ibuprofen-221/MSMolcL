import json
from pathlib import Path

import core.config as app_cfg
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from core.auth import get_current_user
from core.memory_store import set_custom_lib_cache
from core.response import error_response, success_response
from models.user import User
from util.file_utils import get_user_file_path_readonly

candidates_choose_router = APIRouter(prefix="/api", tags=["候选池选择"])

DB_ROOT = Path(app_cfg.retrieve_database_root)
DEFAULT_DATABASES = ["pubchem"]


def _discover_available_databases() -> list[str]:
    if not DB_ROOT.exists() or not DB_ROOT.is_dir():
        return []
    names: set[str] = set()
    for parquet_path in DB_ROOT.rglob("*.parquet"):
        parent_name = parquet_path.parent.name.strip()
        if parent_name:
            names.add(parent_name)
    return sorted(names)


def _parse_databases(raw: str | None, available: list[str]) -> list[str]:
    if not available:
        return []

    if raw is None or not str(raw).strip():
        defaults = [db for db in DEFAULT_DATABASES if db in available]
        return defaults or [available[0]]

    text = str(raw).strip()
    parsed: list[str] = []

    if text.startswith("["):
        try:
            values = json.loads(text)
            if isinstance(values, list):
                parsed = [str(v).strip() for v in values if str(v).strip()]
        except json.JSONDecodeError:
            parsed = []
    else:
        parsed = [item.strip() for item in text.split(",") if item.strip()]

    if not parsed:
        defaults = [db for db in DEFAULT_DATABASES if db in available]
        return defaults or [available[0]]

    parsed = list(dict.fromkeys(parsed))
    invalid = [db for db in parsed if db not in available]
    if invalid:
        raise ValueError(f"databases 包含无效项: {', '.join(invalid)}")
    return parsed


@candidates_choose_router.get("/candidate_databases", status_code=status.HTTP_200_OK)
async def candidate_databases(
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    available = _discover_available_databases()
    defaults = [db for db in DEFAULT_DATABASES if db in available]
    if not defaults and available:
        defaults = [available[0]]
    return success_response(
        message="候选数据库列表",
        data={
            "available_databases": available,
            "default_databases": defaults,
        },
    )


@candidates_choose_router.post("/candidates_choose", status_code=status.HTTP_200_OK)
async def candidates_choose(
    search_type: str = Form(...),
    ion_mode: str = Form(default="pos"),
    ppm_range: float | None = Form(default=None),
    databases: str | None = Form(default=None),
    custom_lib_file: UploadFile | None = File(default=None),
    task_id: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    """候选池选择接口：校验参数并输出检索所需配置。"""
    allowed_types = {"pubchem", "custom"}
    if search_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content=error_response(message="search_type 仅支持 pubchem/custom", code=400),
        )

    ion_mode = (ion_mode or "pos").strip().lower()
    if ion_mode not in {"pos", "neg"}:
        return JSONResponse(
            status_code=400,
            content=error_response(message="ion_mode 仅支持 pos/neg", code=400),
        )

    available_databases = _discover_available_databases()
    try:
        selected_databases = _parse_databases(databases, available_databases)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=error_response(message=str(exc), code=400),
        )

    statas_path = get_user_file_path_readonly(current_user.data_path, task_id, "statas.json")
    fragtrees_path = get_user_file_path_readonly(current_user.data_path, task_id, "valid_pairs_fragtrees.json")
    spectra_path = get_user_file_path_readonly(current_user.data_path, task_id, "valid_pairs_spectra.mgf")

    task_dir = statas_path.parent
    if not task_dir.exists():
        return JSONResponse(
            status_code=400,
            content=error_response(message="任务ID不存在", code=400),
        )

    for file_path in [statas_path, fragtrees_path, spectra_path]:
        if not file_path.is_file():
            return JSONResponse(
                status_code=400,
                content=error_response(message=f"任务文件缺失: {file_path.name}", code=400),
            )

    custom_lib_file_name = None
    if search_type == "pubchem" and ppm_range is None:
        return JSONResponse(
            status_code=400,
            content=error_response(message="pubchem 模式必须传入 ppm_range", code=400),
        )

    if search_type == "custom":
        if custom_lib_file is None:
            return JSONResponse(
                status_code=400,
                content=error_response(message="custom 模式必须上传 custom_lib_file", code=400),
            )

        ext = Path(custom_lib_file.filename or "").suffix.lower()
        if ext != ".txt":
            return JSONResponse(
                status_code=400,
                content=error_response(message="custom_lib_file 仅支持 txt 格式", code=400),
            )

        content = await custom_lib_file.read()
        smiles_list = [line.strip() for line in content.decode("utf-8").splitlines() if line.strip()]
        smiles_list = list(dict.fromkeys(smiles_list))

        if not smiles_list:
            return JSONResponse(
                status_code=400,
                content=error_response(message="custom_lib_file 内容为空", code=400),
            )

        custom_lib_file_name = custom_lib_file.filename or "custom_lib.txt"
        set_custom_lib_cache(custom_lib_file_name, smiles_list)

    data = {
        "task_id": task_id,
        "search_type": search_type,
        "ion_mode": ion_mode,
        "ppm_range": ppm_range,
        "databases": selected_databases,
        "available_databases": available_databases,
        "custom_lib_file_name": custom_lib_file_name,
        "files": {
            "statas_path": str(Path(statas_path).resolve()),
            "fragtrees_path": str(Path(fragtrees_path).resolve()),
            "spectra_path": str(Path(spectra_path).resolve()),
        },
    }

    return success_response(message="候选池选择完成", data=data)
