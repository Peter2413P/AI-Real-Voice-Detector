import io
import numpy as np
import librosa
import joblib
import base64
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Voice AI Detector")

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# CONFIG
# -----------------------
API_KEY = os.getenv("API_KEY")
MODEL_PATH = "voice_auth_model.pkl"
SCALER_PATH = "scaler.pkl"

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# -----------------------
# FEATURE EXTRACTION
# -----------------------
def extract_features_v2(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=4.0)

    if len(y) < 22050 * 4:
        y = np.pad(y, (0, 22050 * 4 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(centroid.T, axis=0)
    ]).reshape(1, -1)


# -----------------------
# HEALTH ROUTE (UptimeRobot)
# -----------------------
@app.get("/")
def home():
    return {"status": "Alive and Kicking!"}


# -----------------------
# MAIN API
# -----------------------
@app.post("/api/voice-detection")
def detect_voice(req: AudioRequest, x_api_key: str = Header(None)):

    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    audio_bytes = base64.b64decode(req.audioBase64)
    X = extract_features_v2(audio_bytes)

    if scaler:
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    conf = float(np.max(model.predict_proba(X)[0]))

    label = "AI_GENERATED" if pred == 1 else "HUMAN"

    return {
        "classification": label,
        "confidenceScore": round(conf, 2),
        "explanation": "Analysis based on MFCC Texture, Delta, and Spectral Centroid."
    }
