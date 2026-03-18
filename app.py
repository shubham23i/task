from Emotion_State_Prediction.src.logger import logging
import sys
from Emotion_State_Prediction.src.exception import customexception

if __name__=='__main__':
    try:
        a=1/0
    except Exception as e:
        logging.info("hi")
        raise customexception(e,sys)
        