"""
Global configuration for audio processing and model training.
These parameters must be fixed for both training and inference.
"""

SAMPLE_RATE = 16000  # 16 kHz as specified
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13
N_LFCC = 13
N_MELS = 128

# Model path relative to project root
MODEL_SAVE_PATH = "app/models/voice_ai_model.joblib"

# Feature ordering for the vector (to ensure consistency)
# 1. MFCC (13 mean + 13 std = 26)
# 2. LFCC (13 mean + 13 std = 26)
# 3. Log Mel-Spec (128 mean + 128 std = 256) -> wait, req says "summary (mean + std across time)"
#    Req says: "Log Mel-Spectrogram summary (mean + std across time)" 
#    If I mean+std across time for each band, it's 2 * N_MELS.
# 4. Spectral Centroid (mean + std = 2)
# 5. Spectral Rolloff (mean + std = 2)
# 6. CQT summary (mean + std = 168? or just global mean/std) -> let's do mean/std across its bands too.
# 7. ZCR (mean + std = 2)
# 8. STE (mean + std = 2)
# 9. F0 (mean + var = 2)
# 10. Jitter (1)
# 11. Shimmer (1)
# 12. Pause stats (mean_dur, total_silence_ratio = 2)
