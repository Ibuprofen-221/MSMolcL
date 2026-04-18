from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.auth import get_current_user
from core.response import success_response
from models.user import User
from services.spec_visible import get_single_spectrum_plot_by_title
from util.file_utils import get_user_file_path_readonly, resolve_user_data_dir

spectrum_detail_router = APIRouter(prefix="/api/spectrum", tags=["spectrum_detail"])


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


@spectrum_detail_router.get("/plot", status_code=status.HTTP_200_OK)
async def get_spectrum_plot(
    task_id: str = Query(..., description="任务ID"),
    title: str = Query(..., description="谱图title"),
    current_user: User = Depends(get_current_user),
) -> dict:
    if not task_id.strip():
        raise HTTPException(status_code=400, detail="task_id 不能为空")

    if not title.strip():
        raise HTTPException(status_code=400, detail="title 不能为空")

    mgf_path = get_user_file_path_readonly(current_user.data_path, task_id.strip(), "valid_pairs_spectra.mgf", storage="normal")
    resolved = _validate_mgf_path(mgf_path, resolve_user_data_dir(current_user.data_path))

    try:
        payload = get_single_spectrum_plot_by_title(str(resolved), title.strip())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=500, detail="谱图解析失败") from None

    return success_response(data=payload, message="谱图读取成功")
