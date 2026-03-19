import sys
import pickle
import numpy as np

from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception


def load_model(file_path):
    try:
        logging.info(f"Loading model from {file_path}")
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        raise customexception(e, sys)


def predict(model, X):
    try:
        logging.info("Running predictions")
        predictions = model.predict(X)
        logging.info("Predictions completed")
        return predictions
    except Exception as e:
        raise customexception(e, sys)


def predict_with_confidence(model, X):
    try:
        logging.info("Running predictions with confidence")

        preds = model.predict(X)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            confidence = np.max(probs, axis=1)
        else:
            
            confidence = np.ones(len(preds)) * 0.5

        logging.info("Prediction with confidence completed")
        return preds, confidence

    except Exception as e:
        raise customexception(e, sys)