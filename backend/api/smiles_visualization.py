import asyncio
import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from core.auth import get_current_user
from core.config import rate_limit_default_rule, smiles_image_dir
from core.rate_limit import limiter
from core.response import success_response
from models.user import User
from services.visualization import MoleculeVisualizer

smiles_visualization_router = APIRouter(prefix="/api/smiles", tags=["smiles可视化"])

SMILES_BATCH_MAX_ITEMS = 200
SMILES_MAX_LENGTH = 500
SMILES_RENDER_TIMEOUT_SECONDS = 5.0


class SmilesBatchPayload(BaseModel):
    smiles_list: list[str] = Field(
        default_factory=list,
        max_length=SMILES_BATCH_MAX_ITEMS,
        description="待生成图片的smiles列表",
    )


def _smiles_to_filename(smiles: str) -> str:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    return f"{digest}.png"


def _build_result(smiles: str, image_url: str = "", status_text: str = "failed") -> dict:
    return {
        "smiles": smiles,
        "image_url": image_url,
        "status": status_text,
    }


@smiles_visualization_router.post("/visualize", status_code=status.HTTP_200_OK)
@limiter.limit(rate_limit_default_rule)
async def visualize_smiles_batch(
    request: Request,
    payload: SmilesBatchPayload,
    current_user: User = Depends(get_current_user),
):
    _ = request
    _ = current_user
    smiles_image_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for raw_smiles in payload.smiles_list:
        smiles = (raw_smiles or "").strip()
        if not smiles or len(smiles) > SMILES_MAX_LENGTH:
            results.append(_build_result(smiles=raw_smiles or "", status_text="failed"))
            continue

        filename = _smiles_to_filename(smiles)
        target_path: Path = smiles_image_dir / filename
        image_url = f"/smiles_image/{filename}"

        if target_path.exists() and target_path.is_file() and target_path.stat().st_size > 0:
            results.append(_build_result(smiles=smiles, image_url=image_url, status_text="ready"))
            continue

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    MoleculeVisualizer.save_molecule_2d,
                    smiles=smiles,
                    output_path=str(target_path),
                ),
                timeout=SMILES_RENDER_TIMEOUT_SECONDS,
            )
        except Exception:
            results.append(_build_result(smiles=smiles, status_text="failed"))
            continue

        if target_path.exists() and target_path.is_file() and target_path.stat().st_size > 0:
            results.append(_build_result(smiles=smiles, image_url=image_url, status_text="ready"))
        else:
            results.append(_build_result(smiles=smiles, status_text="failed"))

    return success_response(data={"results": results}, message="smiles图片处理完成")
