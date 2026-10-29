import json
from pathlib import Path

import pytest

from app.functions.examples import EXAMPLES_BY_SLUG
from app.volumes.volume import estimate_volume
from tests.helpers import register_and_login

GOLDEN = json.loads((Path(__file__).parent / "golden" / "volumes.json").read_text())


@pytest.mark.parametrize("slug", sorted(GOLDEN))
def test_volume_matches_golden(slug):
    expected = GOLDEN[slug]
    function = EXAMPLES_BY_SLUG[slug].function

    estimate = estimate_volume(function.expression, function.domain, expected["resolution"])

    assert estimate.voxel_volume == pytest.approx(expected["voxel_volume"], rel=1e-6)
    assert estimate.mesh_volume == pytest.approx(expected["mesh_volume"], rel=1e-6)


@pytest.mark.parametrize("slug", ["esfera", "toro"])
def test_voxel_and_mesh_methods_agree(slug):
    function = EXAMPLES_BY_SLUG[slug].function

    estimate = estimate_volume(function.expression, function.domain, 64)

    assert estimate.mesh_volume == pytest.approx(estimate.voxel_volume, rel=0.02)


def test_empty_region_has_zero_volume():
    estimate = estimate_volume("x**2 + y**2 + z**2 + 1", [[-1, 1], [-1, 1], [-1, 1]], 16)

    assert estimate.voxel_volume == 0
    assert estimate.mesh_volume == 0


async def test_volume_endpoint(client):
    _, headers = await register_and_login(client, "ana")
    created = (await client.post("/functions/examples/esfera", headers=headers)).json()

    response = await client.get(f"/functions/{created['id']}/volume", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["mesh_volume"] == pytest.approx(GOLDEN["esfera"]["mesh_volume"], rel=1e-6)
    assert body["domain_volume"] == pytest.approx(27.0)



def test_mesh_volume_of_region_touching_the_border():
    estimate = estimate_volume("x", [[-1, 1], [-1, 1], [-1, 1]], 16)

    assert estimate.mesh_volume == pytest.approx(4.0)