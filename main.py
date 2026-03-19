"""
Grok2API 搴旂敤鍏ュ彛

FastAPI 搴旂敤鍒濆鍖栧拰璺敱娉ㄥ唽
"""

from contextlib import asynccontextmanager
import asyncio
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends

from app.core.auth import verify_api_key
from app.core.config import config, get_config
from app.core.logger import logger, setup_logging
from app.core.exceptions import register_exception_handlers
from app.core.response_middleware import ResponseLoggerMiddleware
from app.api.v1.chat import router as chat_router
from app.api.v1.image import router as image_router
from app.api.v1.files import router as files_router
from app.api.v1.models import router as models_router
from app.api.v1.uploads import router as uploads_router
from app.services.token import get_scheduler


# 鍒濆鍖栨棩蹇?
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_console=False,
    file_logging=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """搴旂敤鐢熷懡鍛ㄦ湡绠＄悊"""

    # 0. 鍏煎杩佺Щ锛氫繚鐣欐棫鐗?data 鐩綍涓殑閰嶇疆/缂撳瓨绛夋暟鎹?
    from app.core.legacy_migration import migrate_legacy_cache_dirs, migrate_legacy_account_settings

    await asyncio.to_thread(migrate_legacy_cache_dirs)

    # 1. 鍔犺浇閰嶇疆锛堝唴閮ㄤ細鑷姩鍚堝苟 defaults + 鍏煎 setting.toml锛?
    await config.ensure_loaded()

    # 1.1 Old account post-migration settings (TOS + BirthDate + NSFW), best-effort
    async def _run_legacy_account_migration():
        try:
            await migrate_legacy_account_settings(concurrency=10)
        except Exception as e:
            logger.warning(f"Legacy account migration failed: {e}")

    asyncio.create_task(_run_legacy_account_migration())

    # 2. 鍚姩鏈嶅姟鏄剧ず
    logger.info("Starting Grok2API...")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version.split()[0]}")

    # 3. 鍚姩 Token 鍒锋柊璋冨害鍣?
    refresh_enabled = get_config("token.auto_refresh", True)
    if refresh_enabled:
        basic_interval = get_config("token.refresh_interval_hours", 8)
        super_interval = get_config("token.super_refresh_interval_hours", 2)
        interval = min(basic_interval, super_interval)
        scheduler = get_scheduler(interval)
        scheduler.start()

    logger.info("Application startup complete.")
    yield

    # 鍏抽棴
    logger.info("Shutting down Grok2API...")

    # Best-effort: stop auto-register to avoid blocking shutdown on background threads.
    try:
        from app.services.register import get_auto_register_manager

        await get_auto_register_manager().stop_job()
    except Exception:
        pass

    from app.core.storage import StorageFactory

    if StorageFactory._instance:
        await StorageFactory._instance.close()

    if refresh_enabled:
        scheduler = get_scheduler()
        scheduler.stop()


def create_app() -> FastAPI:
    """鍒涘缓 FastAPI 搴旂敤"""
    app = FastAPI(
        title="Grok2API",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "Grok2API", "runtime": "python-fastapi"}

    # CORS 閰嶇疆
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 璇锋眰鏃ュ織鍜?ID 涓棿浠?
    app.add_middleware(ResponseLoggerMiddleware)

    @app.middleware("http")
    async def ensure_config_loaded(request: Request, call_next):
        await config.ensure_loaded()
        return await call_next(request)

    # 娉ㄥ唽寮傚父澶勭悊鍣?
    register_exception_handlers(app)

    # 娉ㄥ唽璺敱
    app.include_router(chat_router, prefix="/v1", dependencies=[Depends(verify_api_key)])
    app.include_router(image_router, prefix="/v1", dependencies=[Depends(verify_api_key)])
    app.include_router(models_router, prefix="/v1", dependencies=[Depends(verify_api_key)])
    app.include_router(uploads_router, prefix="/v1", dependencies=[Depends(verify_api_key)])
    app.include_router(files_router, prefix="/v1/files")

    # 闈欐€佹枃浠舵湇鍔?
    #
    # NOTE: Starlette/StaticFiles serves JS as `application/javascript` without a charset.
    # Some browsers/OS locales may then mis-decode UTF-8 and display `????` for Chinese text.
    # Force `charset=utf-8` for JS to avoid mojibake across environments (local/docker).
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent / "app" / "static"
    if static_dir.exists():
        class _UTF8StaticFiles(StaticFiles):
            async def get_response(self, path: str, scope):  # type: ignore[override]
                resp = await super().get_response(path, scope)

                # Starlette uses `mimetypes` which may vary across OS/distros.
                # Ensure UTF-8 decoding for text-like assets to avoid mojibake (`????`) on some locales.
                ctype = (resp.headers.get("content-type", "") or "").strip()
                ctype_l = ctype.lower()
                if "charset=" in ctype_l:
                    return resp

                base = ctype.split(";", 1)[0].strip().lower()
                is_text = base.startswith("text/")
                is_js = base in ("application/javascript", "text/javascript")
                is_json = base == "application/json"
                is_css = base == "text/css"

                # Some servers might respond with empty content-type for 304 etc; fall back by extension.
                if not base:
                    ext = Path(path).suffix.lower()
                    if ext in (".js", ".mjs"):
                        resp.headers["content-type"] = "application/javascript; charset=utf-8"
                    elif ext == ".css":
                        resp.headers["content-type"] = "text/css; charset=utf-8"
                    elif ext in (".html", ".htm"):
                        resp.headers["content-type"] = "text/html; charset=utf-8"
                    return resp

                if is_text or is_js or is_json or is_css:
                    resp.headers["content-type"] = f"{base}; charset=utf-8"
                return resp

        app.mount("/static", _UTF8StaticFiles(directory=static_dir), name="static")

    # 娉ㄥ唽绠＄悊璺敱
    from app.api.v1.admin import router as admin_router

    app.include_router(admin_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    workers = int(os.getenv("SERVER_WORKERS", "1"))

    # 骞冲彴妫€鏌?
    is_windows = platform.system() == "Windows"

    # 鑷姩闄嶇骇
    if is_windows and workers > 1:
        logger.warning(
            f"Windows platform detected. Multiple workers ({workers}) is not supported. "
            "Using single worker instead.",
        )
        workers = 1

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
    )

