from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

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
