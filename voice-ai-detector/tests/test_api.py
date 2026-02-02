from fastapi.testclient import TestClient
from app.main import app
import base64
from unittest.mock import patch, MagicMock
import numpy as np

client = TestClient(app)

# Updated test values to match new implementation
SECRET_API_KEY = "sk_test_123456789"

def test_no_api_key():
    response = client.post("/api/voice-detection", json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "dummy"
    })
    assert response.status_code == 403 # Header(...) without auto_error/False might raise 422 if missing? 
    # snippet says Header(...) which usually means required. FastAPI raises 422 for missing required header.
    # However, existing security logic was returning 403. Let's see what snippet does.
    # def verify_api_key(x_api_key: str = Header(...)): -> Required.
    assert response.status_code == 422 

def test_invalid_api_key():
    response = client.post("/api/voice-detection", headers={"x-api-key": "wrong-key"}, json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "dummy"
    })
    assert response.status_code == 401 # Changed from 403 in snippet

@patch("app.main.load_audio_from_base64")
@patch("app.main.extract_mfcc")
@patch("app.main.classify_voice")
def test_successful_detection(mock_classify, mock_extract, mock_load):
    # Setup mocks
    mock_load.return_value = (np.zeros(16000), 16000)
    mock_extract.return_value = np.zeros(13)
    mock_classify.return_value = ("AI_GENERATED", 0.92, "Strong vocoder artifacts detected")

    response = client.post(
        "/api/voice-detection", 
        headers={"x-api-key": SECRET_API_KEY}, 
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "VGhpcyBpcyBhIHRlc3Q="
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["classification"] == "AI_GENERATED"
    assert data["language"] == "English"
    assert data["confidenceScore"] == 0.92

def test_invalid_input_format():
    response = client.post(
        "/api/voice-detection", 
        headers={"x-api-key": SECRET_API_KEY}, 
        json={
            "language": "French", # Invalid language
            "audioFormat": "mp3",
            "audioBase64": "dummy"
        }
    )
    assert response.status_code == 422 # Validation Error
