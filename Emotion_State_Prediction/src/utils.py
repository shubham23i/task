import pickle
import sys
from src.logger import logging
from src.exception import customexception

def save_object(file_path, obj):
    try:
        logging.info(f"Saving object to {file_path}")
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logging.info("Object saved successfully")
    except Exception as e:
        raise customexception(e, sys)