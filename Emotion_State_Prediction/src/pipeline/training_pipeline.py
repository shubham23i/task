import sys
import numpy as np
from sklearn.model_selection import train_test_split

from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception

from Emotion_State_Prediction.src.data.load_data import load_train_data
from Emotion_State_Prediction.src.data.preprocess import preprocess_train
from Emotion_State_Prediction.src.features.build_features import split_features_target, scale_train_features

from Emotion_State_Prediction.src.models.train import ModelTrainer

def run_training():
    try:
        logging.info("Training pipeline started")

        df = load_train_data()
        df = preprocess_train(df)

        X, y = split_features_target(df, "target")

        X_scaled = scale_train_features(X)

        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        train_array = np.c_[X_train, y_train]
        test_array = np.c_[X_test, y_test]

        model_trainer = ModelTrainer()
        best_score = model_trainer.initiate_model_trainer(
            train_array, test_array
        )

        logging.info(f"Best model score: {best_score}")

    except Exception as e:
        raise customexception(e, sys)