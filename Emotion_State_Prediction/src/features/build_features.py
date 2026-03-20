import sys
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception


def split_features_target(df, target_column="emotional_state"):
    try:
        logging.info("Splitting features and target")
        X = df["journal_text"]
        y = df[target_column]

        logging.info(f"Feature shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")

        return X, y

    except Exception as e:
        raise customexception(e, sys)


def transform_train_features(X_train):
    try:
        logging.info("Transforming training features using TF-IDF")

        os.makedirs("artifacts", exist_ok=True)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_transformed = vectorizer.fit_transform(X_train)

        with open("artifacts/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        logging.info("Vectorizer saved")

        return X_transformed

    except Exception as e:
        raise customexception(e, sys)


def transform_test_features(X_test):
    try:
        logging.info("Transforming test features using saved TF-IDF")

        if not os.path.exists("artifacts/vectorizer.pkl"):
            raise FileNotFoundError("vectorizer.pkl not found. Run training first.")

        with open("artifacts/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        X_transformed = vectorizer.transform(X_test)

        return X_transformed

    except Exception as e:
        raise customexception(e, sys)