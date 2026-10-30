from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.measure import marching_cubes

from app.functions.parser import FloatArray
from app.volumes.sampling import SampledGrid

ISO_LEVEL = 0.0
# Valor "exterior" (f > 0) con el que se rodea la rejilla para cerrar la superficie en el borde.
OUTSIDE_VALUE = 1.0
# Cota para los infinitos (p. ej. 1/x en x = 0): grande, pero lejos del máximo de float64 para que
# la interpolación de marching cubes no desborde al restar dos valores extremos.
INFINITY_BOUND = np.finfo(np.float64).max / 4


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
    # Sin ninguna muestra con f <= 0 no hay región. Si todas están dentro, la región es la caja
    # entera y las tapas del relleno la cierran, así que no se descarta.
    if not np.any(values <= ISO_LEVEL):
        return TriangleMesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))

    # Donde f no está definida (NaN, p. ej. sqrt de un negativo) el punto no cumple f <= 0: se trata
    # como exterior, igual que en el conteo de vóxeles. Los infinitos se acotan para poder interpolar.
    values = np.nan_to_num(values, nan=OUTSIDE_VALUE, posinf=INFINITY_BOUND, neginf=-INFINITY_BOUND)

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
