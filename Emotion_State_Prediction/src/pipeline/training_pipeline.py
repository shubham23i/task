import sys
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception

from Emotion_State_Prediction.src.data.load_data import load_train_data
from Emotion_State_Prediction.src.data.preprocess import preprocess_train

from Emotion_State_Prediction.src.features.build_features import split_features_target
from Emotion_State_Prediction.src.features.build_features import transform_train_features
from Emotion_State_Prediction.src.features.build_features import transform_test_features
from sklearn.preprocessing import LabelEncoder


from Emotion_State_Prediction.src.models.train import ModelTrainer


def run_training():
    try:
        logging.info("Training pipeline started")

        df = load_train_data()
        logging.info(f"Data loaded with shape: {df.shape}")

        df = preprocess_train(df)
        logging.info("Data preprocessing completed")

        X, y = split_features_target(df, target_column="emotional_state")
        print("Dataset size:", len(y))
        print("Unique labels:", set(y))
        print("Sample texts:", X.head())
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        le = LabelEncoder()

        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")

        X_train_transformed = transform_train_features(X_train)
        X_test_transformed = transform_test_features(X_test)

        logging.info("Feature transformation completed")

        X_train_array = X_train_transformed.toarray()
        X_test_array = X_test_transformed.toarray()

        train_array = np.c_[X_train_array, y_train]
        test_array = np.c_[X_test_array, y_test]

        logging.info("Train and test arrays created")

        model_trainer = ModelTrainer()
        best_score = model_trainer.initiate_model_trainer(
            train_array, test_array
        )

        logging.info(f"Best model score: {best_score}")
        logging.info("Training pipeline completed")

    except Exception as e:
        raise customexception(e, sys)