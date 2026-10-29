"""Imprime la tabla de convergencia del volumen para el README.

Uso (desde la raíz del proyecto): uv run python -m scripts.convergencia
"""

from math import pi

from app.functions.examples import EXAMPLES_BY_SLUG
from app.volumes.volume import estimate_volume

ANALYTIC_VOLUMES = {
    "esfera": 4 / 3 * pi,
    "toro": 2 * pi**2 * 1 * 0.25**2,
    "gyroide": (2 * pi) ** 3 / 2,
}
RESOLUTIONS = (32, 64, 128, 256)


def relative_error(value: float, expected: float) -> str:
    return f"{abs(value - expected) / expected * 100:.3g} %"


def main() -> None:
    print(
        "| Función | Valor analítico | n | Volumen por malla | Error malla "
        "| Volumen por vóxeles | Error vóxeles |"
    )
    print("|---|---|---|---|---|---|---|")
    for slug, expected in ANALYTIC_VOLUMES.items():
        function = EXAMPLES_BY_SLUG[slug].function
        for n in RESOLUTIONS:
            estimate = estimate_volume(function.expression, function.domain, n)
            print(
                f"| {function.name} | {expected:.6f} | {n} "
                f"| {estimate.mesh_volume:.6f} | {relative_error(estimate.mesh_volume, expected)} "
                f"| {estimate.voxel_volume:.6f} | {relative_error(estimate.voxel_volume, expected)} |"
            )


if __name__ == "__main__":
    main()