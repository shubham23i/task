import sys
from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception

def clean_data(df):
    try:
        logging.info("Starting data cleaning")
        df = df.drop_duplicates()
        df = df.ffill()
        logging.info("Data cleaning completed")
        return df
    except Exception as e:
        raise customexception(e, sys)

def preprocess_train(df):
    return clean_data(df)

def preprocess_test(df):
    return clean_data(df)