from fastapi import FastAPI, Depends
from app.schemas import VoiceDetectionRequest, VoiceDetectionResponse
from app.security.api_key import verify_api_key
from app.services.audio_utils import load_audio_from_base64
from app.services.feature_extractor import extract_features
from app.services.inference import classify_voice

app = FastAPI(title="AI Voice Detection API")


@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    y, sr = load_audio_from_base64(request.audioBase64)
    features = extract_features(y, sr)

    classification, confidence, explanation = classify_voice(features)

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
