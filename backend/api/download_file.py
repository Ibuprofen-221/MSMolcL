from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse, JSONResponse

from core.auth import get_current_user
from core.response import error_response
from models.user import User
from util.file_utils import get_user_file_path_readonly


download_router = APIRouter(prefix="/api", tags=["文件下载"])

ALLOWED_FILENAMES = {"statas.json", "statas_advanced.json", "valid_pairs_fragtrees.json", "valid_pairs_spectra.mgf"}


@download_router.get("/download-file", status_code=status.HTTP_200_OK)
async def download_file(
    task_id: str = Query(...),
    filename: str = Query(...),
    task_space: str = Query(default="normal", description="任务空间：normal/advanced（advanced兼容映射到normal目录）"),
    current_user: User = Depends(get_current_user),
):
    """下载任务目录中的指定文件。"""
    if filename not in ALLOWED_FILENAMES:
        return JSONResponse(status_code=400, content=error_response(message="不支持的文件类型", code=400))

    if task_space not in {"normal", "advanced"}:
        return JSONResponse(status_code=400, content=error_response(message="task_space 仅支持 normal/advanced", code=400))

    file_path = get_user_file_path_readonly(current_user.data_path, task_id, filename, storage="normal")

    if not file_path.parent.exists():
        return JSONResponse(status_code=400, content=error_response(message="任务ID不存在", code=400))

    if not file_path.is_file():
        return JSONResponse(status_code=400, content=error_response(message="文件不存在", code=400))

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
