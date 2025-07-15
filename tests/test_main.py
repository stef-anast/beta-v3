import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import os

# Set the admin API key as an environment variable BEFORE the app is imported.
# This ensures that main.py reads our test key when it's first loaded.
MOCK_API_KEY = "test-key-for-pytest"
os.environ['ADMIN_API_KEY'] = MOCK_API_KEY

from main import app

# --- Fixtures ---

@pytest.fixture
def client():
    """A TestClient that can be used to make requests to the app."""
    with TestClient(app) as c:
        yield c

# --- Test Cases ---

def test_read_root(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Match Prediction API!"}

@pytest.mark.asyncio
async def test_get_predictions_invalid_date_format(client):
    """Test predictions endpoint with a badly formatted date."""
    response = client.get("/predictions/2023-13-40")  # Invalid date
    assert response.status_code == 400
    assert "Invalid date format" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_predictions_model_not_loaded(client):
    """Test predictions endpoint when the model is not loaded in the app state."""
    # Temporarily modify app state for this test
    original_model_state = app.state.ml_models
    app.state.ml_models = {'best_model': None}

    response = client.get("/predictions/2023-01-01")
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]

    # Restore state to not affect other tests
    app.state.ml_models = original_model_state

def test_admin_endpoint_unauthorized(client):
    """Test that admin endpoints are protected against missing API keys."""
    response = client.post("/admin/clear-data")  # No API key header
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authenticated"

def test_admin_endpoint_wrong_key(client):
    """Test that admin endpoints are protected against incorrect API keys."""
    response = client.post(
        "/admin/clear-data",
        headers={"X-API-Key": "this-is-the-wrong-key"}
    )
    assert response.status_code == 403
    assert "Could not validate credentials" in response.json()["detail"]

@pytest.mark.asyncio
async def test_admin_clear_data_authorized(client):
    """Test the happy path for an authorized admin request, mocking the DB."""
    # Mock the database dependency for this single test
    app.dependency_overrides['get_db'] = lambda: AsyncMock()

    response = client.post(
        "/admin/clear-data",
        headers={"X-API-Key": MOCK_API_KEY}
    )
    assert response.status_code == 200
    assert "All match data has been cleared" in response.json()["message"]

    # Clear the override after the test
    app.dependency_overrides.clear() 