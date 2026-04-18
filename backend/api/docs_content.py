import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from core.config import docs_source_path, docs_structured_cache_path, temp_dir
from core.response import success_response


docs_content_router = APIRouter(prefix="/api", tags=["说明文档"])


def _parse_docs_lines(content: str) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("##"):
            text = line[2:].strip()
            if text:
                parsed.append({"type": "subtitle", "content": text})
            continue

        if line.startswith("#"):
            text = line[1:].strip()
            if text:
                parsed.append({"type": "title", "content": text})
            continue

        parsed.append({"type": "text", "content": line})

    return parsed


def _write_cache_atomically(payload: list[dict[str, str]], target: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target.with_suffix(target.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(target)


@docs_content_router.get("/docs-content", status_code=status.HTTP_200_OK)
async def get_docs_content() -> dict:
    if not docs_source_path.is_file():
        raise HTTPException(status_code=400, detail="说明文档文件不存在")

    try:
        raw = docs_source_path.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(status_code=500, detail="读取说明文档失败") from None

    parsed = _parse_docs_lines(raw)

    try:
        _write_cache_atomically(parsed, docs_structured_cache_path)
    except Exception:
        raise HTTPException(status_code=500, detail="写入说明文档缓存失败") from None

    return success_response(data=parsed, message="说明文档读取成功")
