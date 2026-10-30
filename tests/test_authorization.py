import pytest

from tests.helpers import register_and_login


async def create_function_for_ana(client):
    _, ana_headers = await register_and_login(client, "ana")
    _, bob_headers = await register_and_login(client, "bob")
    created = (await client.post("/functions/examples/esfera", headers=ana_headers)).json()
    return created["id"], ana_headers, bob_headers


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/functions/{id}"),
        ("PATCH", "/functions/{id}"),
        ("GET", "/functions/{id}/evaluate?x=0&y=0&z=0"),
        ("GET", "/functions/{id}/volume"),
        ("GET", "/functions/{id}/mesh.bin"),
    ],
)
async def test_other_user_cannot_access_function(client, method, path):
    function_id, _, bob_headers = await create_function_for_ana(client)
    body = {"name": "robada"} if method == "PATCH" else None

    response = await client.request(method, path.format(id=function_id), headers=bob_headers, json=body)

    assert response.status_code == 404


async def test_patch_from_other_user_does_not_modify_function(client):
    function_id, ana_headers, bob_headers = await create_function_for_ana(client)

    await client.patch(f"/functions/{function_id}", json={"name": "robada"}, headers=bob_headers)
    response = await client.get(f"/functions/{function_id}", headers=ana_headers)

    assert response.json()["name"] == "Esfera"
    assert response.json()["version"] == 1


async def test_owner_can_access_own_function(client):
    function_id, ana_headers, _ = await create_function_for_ana(client)

    response = await client.get(f"/functions/{function_id}", headers=ana_headers)

    assert response.status_code == 200


async def test_function_requires_token(client):
    function_id, _, _ = await create_function_for_ana(client)

    response = await client.get(f"/functions/{function_id}")

    assert response.status_code == 401