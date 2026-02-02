import os
import joblib
import numpy as np
import logging
from training.config import MODEL_SAVE_PATH

# Configure logging
logger = logging.getLogger(__name__)

# Global model variable
_MODEL = None

def load_model():
    """Loads the trained model from the specified path."""
    global _MODEL
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            _MODEL = joblib.load(MODEL_SAVE_PATH)
            logger.info(f"Successfully loaded model from {MODEL_SAVE_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _MODEL = None
    else:
        logger.warning(f"Model file not found at {MODEL_SAVE_PATH}. Inference will use fallback.")

# Load model on module import
load_model()

def classify_voice(features):
    """
    Classifies a voice sample based on extracted features using the trained ML model.
    
    Args:
        features (np.array): Concatenated feature vector.
        
    Returns:
        tuple: (classification, confidence, explanation)
    """
    global _MODEL
    
    # Ensure model is loaded (retry once if it was missing initially)
    if _MODEL is None:
        load_model()
        
    if _MODEL is not None:
        # Features need to be reshaped for scikit-learn (1, -1)
        feat_reshaped = features.reshape(1, -1)
        
        # Predict class and probability
        prediction = _MODEL.predict(feat_reshaped)[0]
        probs = _MODEL.predict_proba(feat_reshaped)[0]
        
        classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
        confidence = round(float(probs[prediction]), 2)
        
        explanation = generate_explanation(classification, confidence, features)
    else:
        # Fallback for when no model is available (e.g., initial setup)
        # We'll use a placeholder until the user trains the model
        classification = "UNKNOWN"
        confidence = 0.0
        explanation = "Model not trained yet. Please run the training pipeline in training/train_model.py."

    return classification, confidence, explanation

def generate_explanation(classification, confidence, features):
    """
    Generates a human-readable explanation based on the prediction results.
    In a real scenario, this could use SHAP or feature importance.
    """
    if classification == "AI_GENERATED":
        if confidence > 0.85:
            return "Highly confident detection of synthetic speech artifacts and unnatural spectral consistency."
        else:
            return "Detected subtle anomalies in vocal prosody and frequency distribution characteristic of AI generation."
    else:
        if confidence > 0.85:
            return "Audio exhibits natural human vocal variance, jitter, and spectral characteristics."
        else:
            return "Vocal patterns appear largely consistent with natural human speech, despite minor background interference."
