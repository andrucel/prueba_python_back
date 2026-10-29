"""Recalcula tests/golden/volumes.json con el código actual.

Uso (desde la raíz del proyecto): uv run python -m scripts.regenerar_golden
"""

import json
from pathlib import Path

from app.functions.examples import EXAMPLES_BY_SLUG
from app.volumes.volume import estimate_volume

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "tests" / "golden" / "volumes.json"
SLUGS = ("esfera", "toro", "gyroide")
RESOLUTION = 64


def main() -> None:
    golden = {}
    for slug in SLUGS:
        function = EXAMPLES_BY_SLUG[slug].function
        estimate = estimate_volume(function.expression, function.domain, RESOLUTION)
        golden[slug] = {
            "resolution": RESOLUTION,
            "voxel_volume": float(f"{estimate.voxel_volume:.10g}"),
            "mesh_volume": float(f"{estimate.mesh_volume:.10g}"),
        }
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden, indent=2))


if __name__ == "__main__":
    main()