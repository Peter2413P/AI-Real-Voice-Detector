import base64
import tempfile
import librosa


def load_audio_from_base64(audio_base64: str):
    audio_bytes = base64.b64decode(audio_base64)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    y, sr = librosa.load(temp_path, sr=None)
    return y, sr
