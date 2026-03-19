import sys
import pandas as pd

from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.exception import customexception

from Emotion_State_Prediction.src.data.load_data import load_test_data
from Emotion_State_Prediction.src.data.preprocess import preprocess_test
from Emotion_State_Prediction.src.features.build_features import scale_test_features
from Emotion_State_Prediction.src.models.predict import load_model, predict_with_confidence


def run_inference():
    try:
        logging.info("Inference pipeline started")

        df = load_test_data()

        logging.info(f"Test data shape: {df.shape}")

        if "id" in df.columns:
            ids = df["id"]
            df = df.drop(columns=["id"])
        else:
            ids = range(len(df))

        df = preprocess_test(df)

        X_scaled = scale_test_features(df)

        model = load_model("artifacts/model.pkl")

        preds, confidence = predict_with_confidence(model, X_scaled)

        output = pd.DataFrame({
            "id": ids,
            "prediction": preds,
            "confidence": confidence
        })

        output.to_csv("artifacts/predictions.csv", index=False)

        logging.info("Predictions saved to artifacts/predictions.csv")

        print("Inference completed successfully")

    except Exception as e:
        raise customexception(e, sys)