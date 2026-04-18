from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.auth import get_current_user
from core.response import success_response
from models.user import User
from services.spec_visible import generate_single_ms2_plot
from util.file_utils import get_user_file_path_readonly, resolve_user_data_dir

spectrum_router = APIRouter(prefix="/api", tags=["谱图可视化"])


def _validate_mgf_path(target: Path, allowed_root: Path) -> Path:
    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="谱图文件不存在") from None

    if not str(resolved).startswith(str(allowed_root.resolve())):
        raise HTTPException(status_code=400, detail="仅允许读取当前用户目录下的谱图文件") from None

    if resolved.name != "valid_pairs_spectra.mgf":
        raise HTTPException(status_code=400, detail="仅允许读取 valid_pairs_spectra.mgf")

    return resolved


@spectrum_router.get("/spectrum/plot", status_code=status.HTTP_200_OK)
async def get_spectrum_plot(
    task_id: str = Query(..., description="任务ID"),
    title: str = Query(..., description="谱图title"),
    current_user: User = Depends(get_current_user),
) -> dict:
    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title不能为空")

    mgf_path = get_user_file_path_readonly(current_user.data_path, task_id, "valid_pairs_spectra.mgf", storage="normal")
    resolved_path = _validate_mgf_path(mgf_path, resolve_user_data_dir(current_user.data_path))

    spectrum_payload = generate_single_ms2_plot(str(resolved_path), title)
    if spectrum_payload is None:
        raise HTTPException(status_code=404, detail="未找到指定title的谱图")

    return success_response(data=spectrum_payload, message="谱图读取成功")
