import librosa
import numpy as np
from training.extract_features import extract_all_features

def extract_features(y, sr):
    """
    Service wrapper for the comprehensive feature extraction logic.
    Extracts spectral and prosodic features as defined in the training pipeline.
    """
    return extract_all_features(y, sr)
