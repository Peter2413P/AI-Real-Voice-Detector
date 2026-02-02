import os
import librosa
import numpy as np
import scipy.fftpack as fftpack
from .config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC, N_LFCC, N_MELS

def extract_all_features(y, sr=SAMPLE_RATE):
    """
    Extracts a fixed-length concatenated feature vector from audio.
    
    Args:
        y: Audio time series.
        sr: Sampling rate.
        
    Returns:
        np.array: Concatenated feature vector.
    """
    # Ensure sr matches config
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    features = []

    # --- Spectral / Frequency-Domain Features ---

    # 1. MFCCs (13 coefficients, mean + std)
    # Why: Captures the shape of the spectral envelope, widely used in speech recognition.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(mfcc, axis=1))
    features.append(np.std(mfcc, axis=1))

    # 2. LFCCs (13 coefficients, mean + std)
    # Why: Linear Frequency Cepstral Coefficients are better at capturing artifacts in high frequencies 
    # where Mel filters are broad. Excellent for deepfake detection.
    lfcc = extract_lfcc_features(y, sr, n_lfcc=N_LFCC)
    features.append(np.mean(lfcc, axis=1))
    features.append(np.std(lfcc, axis=1))

    # 3. Log Mel-Spectrogram (mean + std across time for each band)
    # Why: Provides a detailed representation of energy distribution across frequencies.
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel_spec = librosa.power_to_db(mel_spec)
    features.append(np.mean(log_mel_spec, axis=1))
    features.append(np.std(log_mel_spec, axis=1))

    # 4. Spectral Centroid (mean + std)
    # Why: Indicates where the "center of mass" of the spectrum is (brightness of sound).
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append([np.mean(centroid), np.std(centroid)])

    # 5. Spectral Rolloff (85%) (mean + std)
    # Why: Frequency below which 85% of the spectral energy lies. AI voices often have different high-freq roll-offs.
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)
    features.append([np.mean(rolloff), np.std(rolloff)])

    # 6. Constant-Q Transform (CQT) (mean + std across bins)
    # Why: Pitch-invariant representation that often reveals subtle phase or harmonic artifacts.
    try:
        cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH))
        features.append(np.mean(cqt, axis=1))
        features.append(np.std(cqt, axis=1))
    except (librosa.util.exceptions.ParameterError, ValueError):
        # Audio might be too short for default CQT; pad or use zeros
        n_bins = 84 # Default librosa bins
        features.append(np.zeros(n_bins))
        features.append(np.zeros(n_bins))

    # --- Prosodic / Temporal Features ---

    # 7. Fundamental frequency (F0) (mean + variance)
    # Why: AI voices often lack the natural micro-prosody (pitch jitter) of human speech.
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=HOP_LENGTH)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) > 0:
        features.append([np.mean(f0_clean), np.var(f0_clean)])
    else:
        features.append([0.0, 0.0])

    # 8. Zero-crossing rate (mean + std)
    # Why: Measures noisiness/percussiveness; AI generators sometimes produce unusual high-freq noise.
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    features.append([np.mean(zcr), np.std(zcr)])

    # 9. Short-time energy (mean + std)
    # Why: Measures loudness variation; AI can be too consistent in amplitude or have cyclic energy patterns.
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    features.append([np.mean(rms), np.std(rms)])

    # 10. Jitter proxy (pitch variation)
    # Why: Cycle-to-cycle variation in fundamental frequency. Synthesizers often fail this realism.
    if len(f0_clean) > 1:
        jitter = np.mean(np.abs(np.diff(f0_clean)))
    else:
        jitter = 0.0
    features.append([jitter])

    # 11. Shimmer proxy (energy variation)
    # Why: Cycle-to-cycle variation in amplitude.
    if rms.shape[1] > 1:
        shimmer = np.mean(np.abs(np.diff(rms)))
    else:
        shimmer = 0.0
    features.append([shimmer])

    # 12. Silence / pause duration statistics
    # Why: AI speech often has unnaturally uniform gaps or lacks breath breaks.
    intervals = librosa.effects.split(y, top_db=30) # Split by energy threshold
    if len(intervals) > 0:
        # Calculate intermediate gap durations in samples
        gaps = []
        for i in range(len(intervals) - 1):
            gaps.append(intervals[i+1][0] - intervals[i][1])
        
        mean_pause = np.mean(gaps) / sr if gaps else 0.0
        total_silence_ratio = (len(y) - np.sum(intervals[:, 1] - intervals[:, 0])) / len(y)
        features.append([mean_pause, total_silence_ratio])
    else:
        features.append([0.0, 1.0])

    # Flatten and concatenate
    return np.concatenate([np.atleast_1d(f).flatten() for f in features])

def extract_lfcc_features(y, sr, n_lfcc=13):
    """Helper to extract Linear Frequency Cepstral Coefficients using manual DCT."""
    # Magnitude Spectrogram (bins are already linearly spaced)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    # Log energy
    S_db = librosa.amplitude_to_db(S)
    # DCT to get cepstral coeffs
    lfcc = fftpack.dct(S_db, axis=0, type=2, norm='ortho')[:n_lfcc]
    return lfcc

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        y_test, sr_test = librosa.load(test_file, sr=SAMPLE_RATE)
        feat_vec = extract_all_features(y_test, sr_test)
        print(f"Extraction successful. Vector size: {len(feat_vec)}")
        print(f"Sample features: {feat_vec[:10]}...")
