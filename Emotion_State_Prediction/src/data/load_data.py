import pandas as pd
import sys
from src.logger import logging
from src.exception import customexception

def load_train_data(path="data/train.csv"):
    try:
        logging.info("Loading train dataset")
        df = pd.read_csv(path)
        logging.info(f"Train data shape: {df.shape}")
        return df
    except Exception as e:
        raise customexception(e, sys)

def load_test_data(path="data/test.csv"):
    try:
        logging.info("Loading test dataset")
        df = pd.read_csv(path)
        logging.info(f"Test data shape: {df.shape}")
        return df
    except Exception as e:
        raise customexception(e, sys)