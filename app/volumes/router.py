from typing import Annotated

from fastapi import APIRouter, Query, Response
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.time import now_formatted
from app.functions.dependencies import OwnedFunction
from app.volumes.binary import encode_mesh
from app.volumes.cache import mesh_cache, volume_cache
from app.volumes.mesh import extract_mesh
from app.volumes.sampling import sample_function
from app.volumes.schemas import VolumeResult
from app.volumes.volume import VolumeEstimate, estimate_volume

router = APIRouter(prefix="/functions", tags=["volumes"])
logger = get_logger(__name__)
settings = get_settings()

ResolutionQuery = Annotated[
    int | None,
    Query(
        ge=settings.min_resolution,
        le=settings.max_resolution,
        description="Resolución para este cálculo; por defecto, la guardada en la función",
    ),
]


@router.get("/{function_id}/volume")
async def get_volume(function: OwnedFunction, resolution: ResolutionQuery = None) -> VolumeResult:
    expression, domain = function.expression, function.domain
    resolution = function.resolution if resolution is None else resolution
    # El cálculo es CPU intensivo: se ejecuta en un hilo para no bloquear el event loop, que debe
    # seguir atendiendo otras peticiones mientras tanto.
    estimate: VolumeEstimate = await run_in_threadpool(
        volume_cache.get_or_compute,
        (function.id, function.version, resolution),
        lambda: estimate_volume(expression, domain, resolution),
    )
    logger.info(
        "volume_computed",
        function_id=function.id,
        resolution=resolution,
        mesh_volume=estimate.mesh_volume,
    )
    return VolumeResult(
        function_id=function.id,
        resolution=resolution,
        voxel_volume=estimate.voxel_volume,
        mesh_volume=estimate.mesh_volume,
        domain_volume=estimate.domain_volume,
        generated_at=now_formatted(),
    )


@router.get("/{function_id}/mesh.bin", response_class=Response)
async def get_mesh(function: OwnedFunction) -> Response:

    def build_payload() -> bytes:
        grid = sample_function(function.expression, function.domain, function.resolution)
        mesh = extract_mesh(grid)
        logger.info("mesh_generated", function_id=function.id, faces=len(mesh.faces))
        return encode_mesh(mesh)

    payload = await run_in_threadpool(mesh_cache.get_or_compute, (function.id, function.version), build_payload)
    return Response(content=payload, media_type="application/octet-stream")
