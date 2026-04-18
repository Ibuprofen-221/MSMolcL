from pathlib import Path

from core.config import retrieve_pubchem_ppm
from core.memory_store import get_custom_lib_cache
from services import retrieve


def build_retrieve_job_payload(
    search_type: str,
    ppm_range: float | None,
    custom_lib_file_name: str | None,
    ion_mode: str | None,
    statas_path: str,
    fragtrees_path: str,
    spectra_path: str,
    databases: list[str] | None = None,
) -> dict:
    """构建并校验检索任务参数。"""
    mode_mapping = {
        "pubchem": "pubchem",
        "custom": "custom",
    }

    retrieve_mode = mode_mapping.get(search_type)
    if retrieve_mode is None:
        raise ValueError("search_type 参数不合法")

    normalized_ion_mode = (ion_mode or "pos").strip().lower()
    if normalized_ion_mode not in {"pos", "neg"}:
        raise ValueError("ion_mode 参数不合法，仅支持 pos/neg")

    custom_smiles_list = None
    if retrieve_mode == "custom":
        if not custom_lib_file_name:
            raise ValueError("custom 模式缺少 custom_lib_file_name")

        custom_smiles_list = get_custom_lib_cache(custom_lib_file_name)
        if not custom_smiles_list:
            raise ValueError("未在内存中找到 custom_lib_file 内容，请重新调用候选池选择接口")

    for file_path in [statas_path, fragtrees_path, spectra_path]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    normalized_databases = list(dict.fromkeys([d.strip() for d in (databases or []) if isinstance(d, str) and d.strip()]))
    if retrieve_mode == "pubchem" and not normalized_databases:
        normalized_databases = ["pubchem"]

    return {
        "search_type": retrieve_mode,
        "ion_mode": normalized_ion_mode,
        "ppm_range": ppm_range if ppm_range is not None else retrieve_pubchem_ppm,
        "databases": normalized_databases,
        "custom_smiles_list": custom_smiles_list,
        "statas_json_path": statas_path,
        "fragtrees_json_path": fragtrees_path,
        "spectra_mgf_path": spectra_path,
    }


def execute_retrieve_job(job_payload: dict) -> dict:
    """执行单个检索任务（运行在进程池子进程）。"""
    try:
        result = retrieve.main(**job_payload)
    except SystemExit as exc:
        raise RuntimeError(f"检索执行失败，退出码: {exc.code}") from None

    if not isinstance(result, dict):
        raise RuntimeError("检索服务返回结果异常")

    return result
