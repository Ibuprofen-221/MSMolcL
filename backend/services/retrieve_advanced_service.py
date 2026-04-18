from pathlib import Path

from services import retrieve_advanced


def build_retrieve_advanced_job_payload(
    ion_mode: str | None,
    statas_path: str,
    fragtrees_path: str,
    spectra_path: str,
) -> dict:
    normalized_ion_mode = (ion_mode or "pos").strip().lower()
    if normalized_ion_mode not in {"pos", "neg"}:
        raise ValueError("ion_mode 参数不合法，仅支持 pos/neg")

    for file_path in [statas_path, fragtrees_path, spectra_path]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    return {
        "ion_mode": normalized_ion_mode,
        "statas_json_path": statas_path,
        "fragtrees_json_path": fragtrees_path,
        "spectra_mgf_path": spectra_path,
    }


def execute_retrieve_advanced_job(job_payload: dict) -> dict:
    try:
        result = retrieve_advanced.main(**job_payload)
    except SystemExit as exc:
        raise RuntimeError(f"高级检索执行失败，退出码: {exc.code}") from None

    if not isinstance(result, dict):
        raise RuntimeError("高级检索服务返回结果异常")

    return result
