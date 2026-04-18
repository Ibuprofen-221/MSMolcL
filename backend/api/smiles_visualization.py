import hashlib
from pathlib import Path

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from core.config import smiles_image_dir
from core.response import success_response
from services.visualization import MoleculeVisualizer

smiles_visualization_router = APIRouter(prefix="/api/smiles", tags=["smiles可视化"])


class SmilesBatchPayload(BaseModel):
    smiles_list: list[str] = Field(default_factory=list, description="待生成图片的smiles列表")


def _smiles_to_filename(smiles: str) -> str:
    # 直接使用完整的 SHA-1 哈希值作为文件名，安全且唯一
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    return f"{digest}.png"


def _build_result(smiles: str, image_url: str = "", status_text: str = "failed") -> dict:
    return {
        "smiles": smiles,
        "image_url": image_url,
        "status": status_text,
    }


@smiles_visualization_router.post("/visualize", status_code=status.HTTP_200_OK)
async def visualize_smiles_batch(payload: SmilesBatchPayload):
    smiles_image_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for raw_smiles in payload.smiles_list:
        smiles = (raw_smiles or "").strip()
        if not smiles:
            results.append(_build_result(smiles=raw_smiles or "", status_text="failed"))
            continue

        filename = _smiles_to_filename(smiles)
        target_path: Path = smiles_image_dir / filename
        image_url = f"/smiles_image/{filename}"

        if target_path.exists() and target_path.is_file() and target_path.stat().st_size > 0:
            results.append(_build_result(smiles=smiles, image_url=image_url, status_text="ready"))
            continue

        MoleculeVisualizer.save_molecule_2d(smiles=smiles, output_path=str(target_path))
        if target_path.exists() and target_path.is_file() and target_path.stat().st_size > 0:
            results.append(_build_result(smiles=smiles, image_url=image_url, status_text="ready"))
        else:
            results.append(_build_result(smiles=smiles, status_text="failed"))

    return success_response(data={"results": results}, message="smiles图片处理完成")
