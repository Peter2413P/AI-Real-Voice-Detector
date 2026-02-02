import os
import glob
import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Relative imports from the same package
try:
    from .extract_features import extract_all_features
    from .config import SAMPLE_RATE, MODEL_SAVE_PATH
except ImportError:
    # If running as a script directly
    from extract_features import extract_all_features
    from config import SAMPLE_RATE, MODEL_SAVE_PATH

def train_offline():
    """
    Offline training script logic.
    Loads data from data/real/ and data/ai/, extracts features, trains and saves the model.
    """
    data_dir = "data"
    real_dir = os.path.join(data_dir, "real")
    ai_dir = os.path.join(data_dir, "ai")

    # Ensure directories exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    X = []
    y = []

    print("--- Starting Feature Extraction ---")
    print(f"Searching in: {os.path.abspath(real_dir)}")

    # Load Real voices (label 0)
    real_files = glob.glob(os.path.join(real_dir, "*.mp3")) + glob.glob(os.path.join(real_dir, "*.wav"))
    print(f"Found {len(real_files)} real files.")
    for f in real_files:
        print(f"Processing Real: {os.path.basename(f)}")
        try:
            audio, sr = librosa.load(f, sr=SAMPLE_RATE)
            feats = extract_all_features(audio, sr)
            X.append(feats)
            y.append(0)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Load AI voices (label 1)
    ai_files = glob.glob(os.path.join(ai_dir, "*.mp3")) + glob.glob(os.path.join(ai_dir, "*.wav"))
    for f in ai_files:
        print(f"Processing AI: {os.path.basename(f)}")
        try:
            audio, sr = librosa.load(f, sr=SAMPLE_RATE)
            feats = extract_all_features(audio, sr)
            X.append(feats)
            y.append(1)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if len(X) == 0:
        print("Error: No training data found in data/real/ or data/ai/")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset summary: {X.shape[0]} samples, {X.shape[1]} features.")

    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("--- Training RandomForest Model ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict(X_test)
    print("--- Validation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["HUMAN", "AI_GENERATED"]))

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_offline()
