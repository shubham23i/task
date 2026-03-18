import joblib
import sys
from src.logger import logging
from src.exception import customexception

def load_model(path):
    try:
        logging.info("Loading model")
        return joblib.load(path)
    except Exception as e:
        raise customexception(e, sys)

def predict(model, X):
    try:
        logging.info("Running predictions")
        preds = model.predict(X)
        logging.info("Prediction completed")
        return preds
    except Exception as e:
        raise customexception(e, sys)