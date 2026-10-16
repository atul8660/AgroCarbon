import io
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.predict import CarbonAnalysisResponse
from app.auth.dependencies import get_current_user
from app.api.dependencies import get_db

client = TestClient(app)


def test_read_root():
    response = client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_missing_files_upload():
    """Without auth, the endpoint should reject with 401 before even checking form data."""
    response = client.post("/api/v1/analyze")
    assert response.status_code in (401, 422)


# ─── Mock Pipeline Test ───────────────────────────────────────────────────────

MOCK_ANALYSIS_RESULT = CarbonAnalysisResponse(
    species="Mangifera indica",
    species_confidence=0.92,
    health="Healthy",
    health_confidence=0.88,
    predicted_biomass_kg=85.0,
    carbon_stored_kg=39.95,
    co2_equivalent_kg=146.62,
    estimated_carbon_credit=2.2,
    overall_confidence=0.90,
)


def _make_image_bytes() -> bytes:
    """Returns a minimal JPEG in bytes for upload testing."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), color=(255, 255, 255)).save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Minimal valid JPEG marker sequence
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00"
            b"\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
        )


def test_analyze_mock_pipeline():
    """
    Integration test for POST /api/v1/analyze with mocked pipeline, storage, and auth.
    Uses FastAPI dependency_overrides for proper DI injection (not @patch on the function).
    """
    # --- Build fake dependencies ---
    fake_user = MagicMock()
    fake_user.id = 1
    fake_user.role = "Farmer"

    fake_prediction = MagicMock()
    fake_prediction.id = 42
    fake_prediction.prediction_date = datetime(2026, 7, 21, 10, 0, 0)

    fake_db = MagicMock()
    fake_db.add.return_value = None
    fake_db.commit.return_value = None
    fake_db.refresh.side_effect = lambda obj: setattr(obj, "id", 42) or setattr(obj, "prediction_date", datetime(2026, 7, 21, 10, 0, 0))

    # --- Override FastAPI dependencies via app.dependency_overrides ---
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: fake_db

    # --- Mock the pipeline on app.state ---
    mock_pipeline = MagicMock()
    mock_pipeline.run_pipeline = AsyncMock(return_value=MOCK_ANALYSIS_RESULT)
    app.state.pipeline = mock_pipeline

    try:
        with patch("app.api.endpoints.StorageService") as MockStorage:
            storage_instance = MockStorage.return_value
            storage_instance.upload_full_tree.return_value = "https://s3.example.com/full_tree.jpg"
            storage_instance.upload_closeup.return_value = "https://s3.example.com/closeup.jpg"
            storage_instance.upload_report.return_value = None

            with patch("app.api.endpoints.validate_image", new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = _make_image_bytes()

                img_bytes = _make_image_bytes()
                response = client.post(
                    "/api/v1/analyze",
                    files={
                        "full_tree_image": ("full_tree.jpg", img_bytes, "image/jpeg"),
                        "closeup_image": ("closeup.jpg", img_bytes, "image/jpeg"),
                    },
                )
    finally:
        # Always clean up overrides so other tests are not affected
        app.dependency_overrides.clear()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Verify all core Carbon Assessment fields are present
    required_fields = [
        "species", "species_confidence", "health", "health_confidence",
        "predicted_biomass_kg", "carbon_stored_kg", "co2_equivalent_kg",
        "estimated_carbon_credit", "overall_confidence",
    ]
    for field in required_fields:
        assert field in data, f"Missing field in response: {field}"

    assert data["species"] == "Mangifera indica"
    assert data["carbon_stored_kg"] == pytest.approx(39.95, rel=1e-3)
    assert data["overall_confidence"] == pytest.approx(0.90, rel=1e-3)
