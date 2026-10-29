from dataclasses import dataclass

import numpy as np

from app.volumes.mesh import TriangleMesh, extract_mesh
from app.volumes.sampling import SampledGrid, sample_function


@dataclass(frozen=True)
class VolumeEstimate:
    voxel_volume: float
    mesh_volume: float
    domain_volume: float

def voxel_volume(grid: SampledGrid) -> float:
    """Suma el volumen de la celda de cada muestra con f <= 0.

    La celda de una muestra es el cubo centrado en ella, recortado al dominio: en las caras
    solo la mitad queda dentro, por eso esas muestras pesan 1/2 en cada eje.
    """
    weights = [np.ones(n) for n in grid.shape]
    for axis_weights in weights:
        axis_weights[0] = axis_weights[-1] = 0.5
    inside = (grid.values <= 0).astype(np.float64)
    weighted_count = np.einsum("ijk,i,j,k->", inside, *weights)
    return float(weighted_count * np.prod(grid.spacing))

def mesh_volume(mesh: TriangleMesh) -> float:
    """Volumen encerrado por una malla cerrada (teorema de la divergencia, tetraedros con el origen)."""
    if mesh.is_empty:
        return 0.0
    # El volumen de una malla cerrada no depende de dónde esté; se centra en su centroide para no
    # restar números enormes y casi iguales cuando el dominio está lejos del origen (x ~ 600000).
    vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    v0, v1, v2 = (vertices[mesh.faces[:, i]] for i in range(3))
    # Con la malla cerrada y las normales hacia afuera el resultado es positivo. Un volumen negativo
    # indicaría un error de orientación, así que no se oculta con abs().
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def domain_volume(domain: list[list[float]]) -> float:
    return float(np.prod([upper - lower for lower, upper in domain]))


def estimate_volume(expression: str, domain: list[list[float]], resolution: int) -> VolumeEstimate:
    grid = sample_function(expression, domain, resolution)
    return VolumeEstimate(
        voxel_volume=voxel_volume(grid),
        mesh_volume=mesh_volume(extract_mesh(grid)),
        domain_volume=domain_volume(domain),
    )
