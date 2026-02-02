# Voice AI Detector API

## Project Purpose
A FastAPI-based backend to detect whether an uploaded voice sample is Human or AI-generated. This project focuses on signal artifacts (vocoder imperfections) and currently uses a mock inference pipeline.

## How to Run Locally

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server:**
    ```bash
    uvicorn app.main:app --reload
    ```

3.  **Access the API:**
    Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## API Usage

**Endpoint:** `POST /api/voice-detection`

**Headers:**
- `x-api-key`: `your_secret_key` (Default mock key is likely implemented in `app/security/api_key.py`)

**Request Body (JSON):**
```json
{
  "language": "en",
  "audioFormat": "wav",
  "audioBase64": "UklGRi..."
}
```

**Response:**
```json
{
  "status": "success",
  "language": "en",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Strong vocoder artifacts detected"
}
```

## Future Work
-   **Dataset:** A dataset of human and AI-generated voice samples will be added later.
-   **Training:** The `cnn_model.py` architecture is defined but untrained. Training scripts and model weights will be integrated in future phases.
