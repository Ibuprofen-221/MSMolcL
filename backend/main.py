from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.auth import auth_router
from api.candidates_choose import candidates_choose_router
from api.docs_content import docs_content_router
from api.download_file import download_router
from api.file_upload import file_upload_router
from api.health import health_router
from api.history import history_router
from api.my_data import my_data_router
from api.retrieve import retrieve_router
from api.smiles_visualization import smiles_visualization_router
from api.spectrum import spectrum_router
from api.statas import statas_router
from core.config import (
    backend_host,
    backend_port,
    backend_reload,
    cors_allow_credentials,
    cors_allow_headers,
    cors_allow_methods,
    cors_allow_origin_regex,
    cors_allow_origins,
    smiles_image_dir,
)
from core.db import init_db
from core.rate_limit import limiter
from core.response import error_response
from services.retrieve_advanced_runtime import retrieve_advanced_runtime
from services.retrieve_runtime import retrieve_runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    await retrieve_runtime.startup()
    await retrieve_advanced_runtime.startup()
    try:
        yield
    finally:
        await retrieve_advanced_runtime.shutdown()
        await retrieve_runtime.shutdown()


# 创建 FastAPI 应用实例
app = FastAPI(lifespan=lifespan)

# 注册限流器与中间件
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    detail = str(exc.detail) if getattr(exc, "detail", None) else "请求过于频繁，请稍后重试"
    return JSONResponse(status_code=429, content=error_response(message=detail, code=429))


app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_origin_regex=cors_allow_origin_regex,
    allow_credentials=cors_allow_credentials,
    allow_methods=cors_allow_methods,
    allow_headers=cors_allow_headers,
)

smiles_image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/smiles_image", StaticFiles(directory=str(smiles_image_dir)), name="smiles_image")

# 注册路由
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(my_data_router)
app.include_router(file_upload_router)
app.include_router(candidates_choose_router)
app.include_router(retrieve_router)
app.include_router(history_router)
app.include_router(download_router)
app.include_router(statas_router)
app.include_router(spectrum_router)
app.include_router(docs_content_router)
app.include_router(smiles_visualization_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="main:app", host=backend_host, port=backend_port, reload=backend_reload)
