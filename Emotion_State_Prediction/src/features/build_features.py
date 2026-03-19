import sys
import pickle
from sklearn.preprocessing import StandardScaler

from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception


def split_features_target(df, target_column="emotional_state"):
    try:
        logging.info("Splitting features and target")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logging.info(f"Feature shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")

        return X, y

    except Exception as e:
        raise customexception(e, sys)


def scale_train_features(X_train):
    try:
        logging.info("Scaling training features")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        with open("artifacts/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        logging.info("Scaler saved to artifacts/scaler.pkl")

        return X_scaled

    except Exception as e:
        raise customexception(e, sys)

def scale_test_features(X_test):
    try:
        logging.info("Scaling test features")

        with open("artifacts/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return scaler.transform(X_test)

    except Exception as e:
        raise customexception(e, sys)