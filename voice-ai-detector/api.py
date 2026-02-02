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
from fastapi.responses import HTMLResponse

load_dotenv()

app = FastAPI(title="Voice AI Detector")

# 1. SETUP CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CONFIGURATION
API_KEY = os.getenv("API_KEY")
MODEL_PATH = "voice_auth_model.pkl"
SCALER_PATH = "scaler.pkl"  # <--- CRITICAL: You need this file!

# 3. LOAD MODEL & SCALER
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
else:
    model = None
    print(f"❌ Error: {MODEL_PATH} not found.")

# Try to load scaler (fixes accuracy issues)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Scaler loaded: {SCALER_PATH}")
else:
    scaler = None
    print("⚠️  Warning: No scaler.pkl found. Predictions might be bad.")

class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# 4. CORRECT FEATURE EXTRACTION (27 Features: Mean + Delta + Centroid)
def extract_features_v2(audio_bytes):
    try:
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=4.0)
        
        # Pad if too short
        if len(y) < 22050 * 4:
            y = np.pad(y, (0, 22050 * 4 - len(y)))

        # A. MFCC (13)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # B. Delta (13) - THIS IS WHAT THE MODEL USUALLY WANTS
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta.T, axis=0)

        # C. Centroid (1) - THIS IS WHAT THE MODEL USUALLY WANTS
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid.T, axis=0)

        # Total: 13 + 13 + 1 = 27
        return np.hstack([mfcc_mean, delta_mean, centroid_mean]).reshape(1, -1)
    
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

# 5. DYNAMIC FRONTEND SERVING
@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    # Inject the API key into the HTML
    html_content = html_content.replace("REPLACE_WITH_API_KEY", API_KEY or "")
    return html_content

# 6. ENDPOINT
# Health Check Route (UptimeRobot hits this)
@app.get("/")
def home():
    return {"status": "Alive and Kicking!"}
@app.post("/api/voice-detection")
def detect_voice(req: AudioRequest, x_api_key: str = Header(None)):
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Decode
        audio_bytes = base64.b64decode(req.audioBase64)
        
        # Extract
        X = extract_features_v2(audio_bytes)
        
        if X is None:
             raise HTTPException(status_code=400, detail="Could not extract features")

        # Scale (Crucial step)
        if scaler:
            X = scaler.transform(X)

        # Predict
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        conf = float(np.max(probs))

        # Explicit Label Mapping (Don't trust 'pred' directly)
        # Assuming 1 = AI, 0 = HUMAN. Swap strings if your results are backwards.
        if pred == 1:
            label = "AI_GENERATED"
        else:
            label = "HUMAN"

        return {
            "classification": label,
            "confidenceScore": round(conf, 2),
            "explanation": "Analysis based on MFCC Texture, Delta, and Spectral Centroid."
        }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Predict
    pred = model.predict(X)[0]
    conf = float(np.max(model.predict_proba(X)))

    response = {
        "classification": pred,  # "AI_GENERATED" or "HUMAN"
        "confidenceScore": conf,
        "explanation": "Predicted based on MFCC features + RMS energy."
    }
    return response
