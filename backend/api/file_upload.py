import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from core.auth import get_current_user
from core.config import (
    allowed_json_extensions,
    allowed_mgf_extensions,
    rate_limit_upload_rule,
    upload_max_total_files,
)
from core.exceptions import FileFormatError, FileMissingError, FileSizeError
from core.memory_store import update_processed_cache
from core.rate_limit import limiter
from core.response import success_response
from models.user import User
from services.file_preprocess import main as preprocess_main
from services.history_store import upsert_task_record
from services.sirius_batch_service import SiriusBatchError, submit_mgf_only_batch
from util.file_utils import (
    generate_task_id,
    get_file_extension,
    get_user_file_path,
    get_user_task_dir,
    safe_remove_file,
    save_upload_file,
)

file_upload_router = APIRouter(prefix="/api", tags=["文件上传"])


def _safe_stem(filename: str | None) -> str:
    stem = Path(filename or "").stem.strip()
    if not stem:
        return ""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)


def _normalize_upload_lists(
    file_mgf: UploadFile | None,
    file_json: UploadFile | None,
    files_mgf: list[UploadFile],
    files_json: list[UploadFile],
) -> tuple[list[UploadFile], list[UploadFile]]:
    mgf_list: list[UploadFile] = []
    json_list: list[UploadFile] = []

    if file_mgf is not None:
        mgf_list.append(file_mgf)
    if file_json is not None:
        json_list.append(file_json)

    if files_mgf:
        mgf_list.extend([item for item in files_mgf if item is not None])
    if files_json:
        json_list.extend([item for item in files_json if item is not None])

    return mgf_list, json_list


def _validate_upload_files(mgf_files: list[UploadFile], json_files: list[UploadFile]) -> None:
    if not mgf_files or not json_files:
        raise FileMissingError("必须同时上传mgf/txt文件和json文件")

    total_count = len(mgf_files) + len(json_files)
    if total_count > upload_max_total_files:
        raise FileFormatError(f"单次上传总文件数不能超过{upload_max_total_files}个")

    for file_mgf in mgf_files:
        mgf_ext = get_file_extension(file_mgf.filename)
        if mgf_ext not in allowed_mgf_extensions:
            raise FileFormatError(f"文件 {file_mgf.filename} 仅支持mgf/txt格式")

    for file_json in json_files:
        json_ext = get_file_extension(file_json.filename)
        if json_ext not in allowed_json_extensions:
            raise FileFormatError(f"文件 {file_json.filename} 仅支持json格式")


def _build_stem_map(files: list[UploadFile], file_type: str) -> dict[str, UploadFile]:
    stem_map: dict[str, UploadFile] = {}
    for item in files:
        stem = _safe_stem(item.filename)
        if not stem:
            raise FileFormatError(f"{file_type}文件名不合法: {item.filename}")
        if stem in stem_map:
            raise FileFormatError(f"{file_type}存在同名文件（不含后缀）: {stem}")
        stem_map[stem] = item
    return stem_map


def _validate_mgf_only_files(mgf_files: list[UploadFile]) -> None:
    if not mgf_files:
        raise FileMissingError("至少需要上传一个 mgf/txt 文件")
    if len(mgf_files) > upload_max_total_files:
        raise FileFormatError(f"单次上传总文件数不能超过{upload_max_total_files}个")

    for file_mgf in mgf_files:
        mgf_ext = get_file_extension(file_mgf.filename)
        if mgf_ext not in allowed_mgf_extensions:
            raise FileFormatError(f"文件 {file_mgf.filename} 仅支持mgf/txt格式")


@file_upload_router.post("/upload-files", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_upload_rule)
async def upload_files(
    request: Request,
    file_mgf: UploadFile | None = File(default=None),
    file_json: UploadFile | None = File(default=None),
    files_mgf: list[UploadFile] = File(default=[]),
    files_json: list[UploadFile] = File(default=[]),
    current_user: User = Depends(get_current_user),
) -> dict:
    """支持批次上传：按同名(mgf/json)配对预处理，输出统一 task 结果。"""
    _ = request
    task_id = generate_task_id()
    saved_temp_files: list[Path] = []

    all_upload_refs: list[UploadFile] = []
    if file_mgf is not None:
        all_upload_refs.append(file_mgf)
    if file_json is not None:
        all_upload_refs.append(file_json)
    if files_mgf:
        all_upload_refs.extend(files_mgf)
    if files_json:
        all_upload_refs.extend(files_json)

    try:
        mgf_list, json_list = _normalize_upload_lists(file_mgf, file_json, files_mgf, files_json)

        # 新增分支：仅上传 mgf/txt，异步走 Sirius 队列生成 json
        if not json_list:
            _validate_mgf_only_files(mgf_list)
            mgf_stem_map = _build_stem_map(mgf_list, "mgf/txt")
            task_dir = get_user_task_dir(current_user.data_path, task_id, create=True)

            mgf_entries: list[dict] = []
            for idx, stem in enumerate(sorted(mgf_stem_map.keys()), start=1):
                mgf_file = mgf_stem_map[stem]
                mgf_ext = get_file_extension(mgf_file.filename)
                unique_suffix = str(int(time.time() * 1000))
                pair_dir = task_dir / f"pair_{idx:03d}_{stem}"
                pair_dir.mkdir(parents=True, exist_ok=True)

                saved_mgf_path = get_user_file_path(
                    current_user.data_path,
                    task_id,
                    f"upload_mgf_{stem}_{unique_suffix}.{mgf_ext}",
                    create=True,
                )
                await save_upload_file(upload_file=mgf_file, target_path=saved_mgf_path)

                mgf_entries.append(
                    {
                        "pair_key": stem,
                        "source_mgf_name": mgf_file.filename,
                        "mgf_path": str(saved_mgf_path),
                        "pair_dir": str(pair_dir),
                    }
                )

            submit_data = submit_mgf_only_batch(
                user_data_path=current_user.data_path,
                task_id=task_id,
                mgf_entries=mgf_entries,
            )
            upsert_task_record(
                task_id=task_id,
                user_data_path=current_user.data_path,
                normal_status="pending",
                advanced_status="pending",
            )
            return success_response(
                data={
                    **submit_data,
                    "batch_summary": {
                        "paired_count": 0,
                        "unmatched_mgf": [],
                        "unmatched_json": [],
                    },
                    "poll_endpoint": "/api/upload-files/status",
                },
                message="mgf文件上传成功，已进入Sirius队列处理中",
            )

        # 旧分支：mgf + json 同步预处理（保持原逻辑）
        _validate_upload_files(mgf_list, json_list)

        mgf_stem_map = _build_stem_map(mgf_list, "mgf/txt")
        json_stem_map = _build_stem_map(json_list, "json")

        paired_stems = sorted(set(mgf_stem_map.keys()) & set(json_stem_map.keys()))
        unmatched_mgf = sorted(set(mgf_stem_map.keys()) - set(json_stem_map.keys()))
        unmatched_json = sorted(set(json_stem_map.keys()) - set(mgf_stem_map.keys()))

        if not paired_stems:
            raise FileFormatError("未发现同名mgf/json配对文件")

        task_dir = get_user_task_dir(current_user.data_path, task_id, create=True)

        merged_root_info: list[dict] = []
        merged_valid_pairs: list[str] = []
        merged_fragtrees: dict = {}
        merged_spectra_text_parts: list[str] = []
        batch_items: list[dict] = []

        for idx, stem in enumerate(paired_stems, start=1):
            mgf_file = mgf_stem_map[stem]
            json_file = json_stem_map[stem]

            mgf_ext = get_file_extension(mgf_file.filename)
            json_ext = get_file_extension(json_file.filename)
            unique_suffix = str(int(time.time() * 1000))

            saved_mgf_path = get_user_file_path(
                current_user.data_path,
                task_id,
                f"upload_mgf_{stem}_{unique_suffix}.{mgf_ext}",
                create=True,
            )
            saved_json_path = get_user_file_path(
                current_user.data_path,
                task_id,
                f"upload_json_{stem}_{unique_suffix}.{json_ext}",
                create=True,
            )

            await save_upload_file(upload_file=mgf_file, target_path=saved_mgf_path)
            await save_upload_file(upload_file=json_file, target_path=saved_json_path)
            saved_temp_files.extend([saved_mgf_path, saved_json_path])

            pair_dir = task_dir / f"pair_{idx:03d}_{stem}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            preprocess_main(str(saved_mgf_path), str(saved_json_path), output_base_dir=pair_dir)

            pair_statas_path = pair_dir / "statas.json"
            pair_fragtrees_path = pair_dir / "valid_pairs_fragtrees.json"
            pair_spectra_path = pair_dir / "valid_pairs_spectra.mgf"

            if not pair_statas_path.exists() or not pair_fragtrees_path.exists() or not pair_spectra_path.exists():
                raise HTTPException(status_code=500, detail=f"配对文件 {stem} 处理结果不完整")

            pair_statas = json.loads(pair_statas_path.read_text(encoding="utf-8"))
            pair_fragtrees = json.loads(pair_fragtrees_path.read_text(encoding="utf-8"))
            pair_spectra_text = pair_spectra_path.read_text(encoding="utf-8")

            pair_root_info = pair_statas.get("碎裂树文件统计", {}).get("有效碎裂树根节点信息", [])
            pair_valid_pairs = pair_statas.get("最终有效对", [])

            merged_root_info.extend(pair_root_info)
            merged_valid_pairs.extend(pair_valid_pairs)
            merged_fragtrees.update(pair_fragtrees)
            if pair_spectra_text.strip():
                merged_spectra_text_parts.append(pair_spectra_text.strip())

            batch_items.append(
                {
                    "pair_key": stem,
                    "pair_dir": pair_dir.name,
                    "source_files": {
                        "mgf": mgf_file.filename,
                        "json": json_file.filename,
                    },
                    "output_files": {
                        "statas": str(pair_statas_path.resolve()),
                        "fragtrees": str(pair_fragtrees_path.resolve()),
                        "spectra": str(pair_spectra_path.resolve()),
                    },
                    "valid_pairs_count": len(pair_valid_pairs),
                }
            )

        statas_file = get_user_file_path(current_user.data_path, task_id, "statas.json")
        fragtrees_file = get_user_file_path(current_user.data_path, task_id, "valid_pairs_fragtrees.json")
        spectra_file = get_user_file_path(current_user.data_path, task_id, "valid_pairs_spectra.mgf")

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
            "批次文件统计": {
                "mgf上传数": len(mgf_list),
                "json上传数": len(json_list),
                "配对成功数": len(paired_stems),
                "未配对mgf": unmatched_mgf,
                "未配对json": unmatched_json,
            },
            "批次文件": batch_items,
        }

        statas_file.write_text(json.dumps(merged_statas, ensure_ascii=False, indent=2), encoding="utf-8")
        fragtrees_file.write_text(json.dumps(merged_fragtrees, ensure_ascii=False, indent=2), encoding="utf-8")
        spectra_file.write_text("\n\n".join(merged_spectra_text_parts) + ("\n" if merged_spectra_text_parts else ""), encoding="utf-8")

        update_processed_cache(
            spectra_content=spectra_file.read_text(encoding="utf-8"),
            fragtrees_content=merged_fragtrees,
            statas=merged_statas,
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
                "statas": merged_statas,
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
                "batch_summary": {
                    "paired_count": len(paired_stems),
                    "unmatched_mgf": unmatched_mgf,
                    "unmatched_json": unmatched_json,
                },
            },
            message="批次文件上传并处理成功",
        )
    except (FileMissingError, FileFormatError, FileSizeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except SiriusBatchError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="文件处理失败") from None
    finally:
        for upload_ref in all_upload_refs:
            try:
                await upload_ref.close()
            except Exception:
                pass

        for file_path in saved_temp_files:
            safe_remove_file(file_path)


@file_upload_router.post("/upload-files-advanced", status_code=status.HTTP_400_BAD_REQUEST)
async def upload_files_advanced() -> dict:
    """高级检索改为 task_id 模式，已下线文件上传入口。"""
    raise HTTPException(status_code=400, detail="高级检索不再支持上传文件，请在高级检索页面输入任务ID")
