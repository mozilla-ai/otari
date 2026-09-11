from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get(f"{API_ROOT}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_liveness(client: TestClient) -> None:
    """Test liveness probe endpoint."""
    response = client.get(f"{API_ROOT}/health/liveness")
    assert response.status_code == 200
    assert response.text == '"I\'m alive!"'


def test_health_readiness(client: TestClient) -> None:
    """Test readiness probe endpoint."""
    response = client.get(f"{API_ROOT}/health/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "version" in data
