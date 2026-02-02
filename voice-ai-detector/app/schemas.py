from pydantic import BaseModel, Field
from typing import Literal


class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audioFormat: Literal["mp3"]
    audioBase64: str = Field(..., description="Base64 encoded MP3 audio")


class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float
    explanation: str
