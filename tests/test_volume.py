import json
from math import pi
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

    

def test_voxel_volume_of_region_touching_the_border():
    estimate = estimate_volume("x", [[-1, 1], [-1, 1], [-1, 1]], 16)

    assert estimate.voxel_volume == pytest.approx(4.0)

    

def test_mesh_volume_does_not_depend_on_position():
    near = estimate_volume("x**2 + y**2 + z**2 - 1", [[-1.5, 1.5]] * 3, 64)
    far = estimate_volume(
        "(x - 600000)**2 + (y - 600000)**2 + (z - 600000)**2 - 1", [[599998.5, 600001.5]] * 3, 64
    )

    assert far.mesh_volume == pytest.approx(near.mesh_volume, rel=1e-6)

    

ANALYTIC_VOLUMES = {
    "esfera": 4 / 3 * pi,
    "toro": 2 * pi**2 * 1 * 0.25**2,
    "gyroide": (2 * pi) ** 3 / 2,
}


@pytest.mark.parametrize("slug", sorted(ANALYTIC_VOLUMES))
def test_volumes_match_analytic_value(slug):
    function = EXAMPLES_BY_SLUG[slug].function

    estimate = estimate_volume(function.expression, function.domain, 128)

    assert estimate.mesh_volume == pytest.approx(ANALYTIC_VOLUMES[slug], rel=0.005)
    assert estimate.voxel_volume == pytest.approx(ANALYTIC_VOLUMES[slug], rel=0.005)


@pytest.mark.parametrize("slug", ["esfera", "toro"])
def test_mesh_error_decreases_with_resolution(slug):
    function = EXAMPLES_BY_SLUG[slug].function
    expected = ANALYTIC_VOLUMES[slug]

    errors = [
        abs(estimate_volume(function.expression, function.domain, n).mesh_volume - expected)
        for n in (32, 64, 128)
    ]

    assert errors[0] > errors[1] > errors[2]
    

async def test_volume_and_mesh_are_recomputed_after_update(client):
    _, headers = await register_and_login(client, "ana")
    created = (await client.post("/functions/examples/esfera", headers=headers)).json()
    volume_url = f"/functions/{created['id']}/volume"
    mesh_url = f"/functions/{created['id']}/mesh.bin"
    volume_before = (await client.get(volume_url, headers=headers)).json()
    mesh_before = (await client.get(mesh_url, headers=headers)).content

    await client.patch(
        f"/functions/{created['id']}", json={"expression": "x**2 + y**2 + z**2 - 0.25"}, headers=headers
    )
    volume_after = (await client.get(volume_url, headers=headers)).json()
    mesh_after = (await client.get(mesh_url, headers=headers)).content

    assert volume_after["mesh_volume"] == pytest.approx(volume_before["mesh_volume"] / 8, rel=0.05)
    assert mesh_after != mesh_before
    

def test_undefined_points_count_as_outside():
    estimate = estimate_volume("sqrt(x) - 1", [[-2, 2]] * 3, 64)

    assert estimate.mesh_volume == pytest.approx(16.0, rel=0.05)
    assert estimate.voxel_volume == pytest.approx(16.0, rel=0.05)

    

def test_region_filling_the_whole_domain():
    estimate = estimate_volume("x - 5", [[-2, 2]] * 3, 16)

    assert estimate.mesh_volume == pytest.approx(64.0)
    assert estimate.voxel_volume == pytest.approx(64.0)