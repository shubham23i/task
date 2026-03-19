import pandas as pd
import sys
from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception

def load_train_data(path="Emotion_State_Prediction/data/raw/train.csv"):
    try:
        logging.info("Loading train dataset")
        df = pd.read_csv(path)
        logging.info(f"Train data shape: {df.shape}")
        return df
    except Exception as e:
        raise customexception(e, sys)

def load_test_data(path="Emotion_State_Prediction/data/raw/test.csv"):
    try:
        logging.info("Loading test dataset")
        df = pd.read_csv(path)
        logging.info(f"Test data shape: {df.shape}")
        return df
    except Exception as e:
        raise customexception(e, sys)