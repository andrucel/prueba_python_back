from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.measure import marching_cubes

from app.functions.parser import FloatArray
from app.volumes.sampling import SampledGrid

ISO_LEVEL = 0.0
# Valor "exterior" (f > 0) con el que se rodea la rejilla para cerrar la superficie en el borde.
OUTSIDE_VALUE = 1.0


@dataclass(frozen=True)
class TriangleMesh:
    vertices: FloatArray
    faces: NDArray[np.int64]

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0


def extract_mesh(grid: SampledGrid) -> TriangleMesh:
    """Extrae la superficie f = 0 con marching cubes."""
    values = grid.values
    if values.min() > ISO_LEVEL or values.max() < ISO_LEVEL:
        return TriangleMesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))

    # Si la región toca el borde del dominio, la superficie f = 0 queda abierta y el volumen por
    # divergencia no tiene sentido. Se rodea la rejilla con una capa exterior para que marching cubes
    # genere las "tapas" y la malla quede cerrada.
    padded = np.pad(values, 1, constant_values=OUTSIDE_VALUE)
    # f < 0 en el interior: "descent" orienta las normales hacia el exterior.
    vertices, faces, _, _ = marching_cubes(
        padded, level=ISO_LEVEL, spacing=grid.spacing, gradient_direction="descent"
    )
    spacing = np.asarray(grid.spacing)
    # Se descuenta la capa de relleno y las tapas, que caen entre la última muestra y el relleno,
    # se llevan exactamente a las caras del dominio.
    extent = spacing * (np.asarray(values.shape) - 1)
    vertices = np.clip(vertices - spacing, 0.0, extent)
    vertices += np.asarray(grid.origin)
    return TriangleMesh(vertices=vertices, faces=faces.astype(np.int64))
