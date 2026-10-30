import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestLoggingMiddleware, add_cors
from app.db.session import engine, init_db
from app.functions.examples_router import router as examples_router
from app.functions.parser import ExpressionError
from app.functions.router import router as functions_router
from app.users.auth_router import router as auth_router
from app.users.router import router as users_router
from app.volumes.router import router as volumes_router

STATIC_DIR = Path(__file__).parent / "static"

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_db()
    logger.info("app_started", environment=settings.environment)
    yield
    await engine.dispose()
    logger.info("app_stopped")


app = FastAPI(title=settings.name, version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)
add_cors(app, settings.cors_origins)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(examples_router)
app.include_router(functions_router)
app.include_router(volumes_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ExpressionError)
async def expression_error_handler(request: Request, error: ExpressionError) -> JSONResponse:
    # Una función guardada antes de la validación actual puede tener una expresión que ya no se
    # acepta: es un problema de los datos de entrada (422), no un fallo del servidor (500).
    logger.warning("invalid_stored_expression", path=request.url.path, error=str(error))
    return JSONResponse(status_code=422, content={"detail": f"La expresión guardada no es válida: {error}"})


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/viewer", include_in_schema=False)
async def viewer() -> FileResponse:
    return FileResponse(STATIC_DIR / "viewer" / "index.html")


@lru_cache
def render_panel(api_url: str) -> str:
    # Se lee una sola vez por URL: así no se repite la lectura del disco en cada petición.
    html = (STATIC_DIR / "panel" / "index.html").read_text(encoding="utf-8")
    # json.dumps produce un literal de JavaScript válido; escapar "<" impide cerrar el <script>.
    literal = json.dumps(api_url).replace("<", "\\u003c")
    return html.replace("__PANEL_API_URL__", literal)


@app.get("/panel", include_in_schema=False)
async def panel() -> HTMLResponse:
    return HTMLResponse(render_panel(settings.panel_api_url))
