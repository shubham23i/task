import sys
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle

from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception

from Emotion_State_Prediction.src.data.load_data import load_train_data
from Emotion_State_Prediction.src.data.preprocess import preprocess_train

from Emotion_State_Prediction.src.features.build_features import (
    split_features_target,
    transform_train_features,
    transform_test_features,
)

from sklearn.preprocessing import LabelEncoder
from Emotion_State_Prediction.src.models.train import ModelTrainer


def run_training():
    try:
        df = load_train_data()
        df = preprocess_train(df)

        X, y = split_features_target(df)
        y_intensity = df["intensity"]

        X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
            X, y, y_intensity, test_size=0.2, random_state=42, stratify=y
        )

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        X_train_transformed = transform_train_features(X_train)
        X_test_transformed = transform_test_features(X_test)

        X_train_array = X_train_transformed.toarray()
        X_test_array = X_test_transformed.toarray()

        train_array = np.c_[X_train_array, y_train]
        test_array = np.c_[X_test_array, y_test]

        model_trainer = ModelTrainer()
        best_score = model_trainer.initiate_model_trainer(
            train_array, test_array
        )

        model_trainer.train_intensity_model(
            X_train_array, X_test_array, y_train_int, y_test_int
        )

        return best_score

    except Exception as e:
        raise customexception(e, sys)