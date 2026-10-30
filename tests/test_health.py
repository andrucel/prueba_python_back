from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app import main
from app.core.middleware import add_cors


async def test_health_returns_ok(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_propagates_request_id(client):
    response = await client.get("/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["X-Request-ID"] == "abc123"


async def test_viewer_is_served(client):
    response = await client.get("/viewer")

    assert response.status_code == 200
    assert "Visor de volúmenes" in response.text


async def test_cors_allows_only_configured_origins():
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    add_cors(app, ["http://localhost:5173"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        allowed = await client.get("/ping", headers={"Origin": "http://localhost:5173"})
        other = await client.get("/ping", headers={"Origin": "http://otro-sitio.example"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in other.headers


async def test_panel_is_served_with_the_configured_api_url(client, monkeypatch):
    monkeypatch.setattr(main.settings, "panel_api_url", "http://api.example:8000")

    response = await client.get("/panel")

    assert response.status_code == 200
    assert 'const API_URL = "http://api.example:8000";' in response.text
    for test_id in [
        "login-username",
        "login-password",
        "login-submit",
        "function-list",
        "volume-mesh",
        "volume-voxel",
        "volume-domain",
        "volume-difference",
        "surface-view",
        "convergence-table",
        "convergence-mesh",
        "convergence-voxel",
        "error-message",
    ]:
        assert f'data-testid="{test_id}"' in response.text


async def test_panel_uses_same_origin_by_default(client, monkeypatch):
    monkeypatch.setattr(main.settings, "panel_api_url", "")

    response = await client.get("/panel")

    assert 'const API_URL = "";' in response.text
    assert "__PANEL_API_URL__" not in response.text
