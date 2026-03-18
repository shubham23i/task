import pandas as pd
import sys
from src.logger import logging
from src.exception import customexception

from src.data.load_data import load_test_data
from src.data.preprocess import preprocess_test
from src.features.build_features import scale_test_features
from src.models.predict import load_model, predict

def run_inference():
    try:
        logging.info("Inference pipeline started")

        df = load_test_data()

        ids = df["id"] if "id" in df.columns else None

        if "target" in df.columns:
            df = df.drop(columns=["target"])

        df = preprocess_test(df)
        X_scaled = scale_test_features(df)

        model = load_model("artifacts/model.pkl")
        preds = predict(model, X_scaled)

        output = pd.DataFrame({
            "id": ids if ids is not None else range(len(preds)),
            "prediction": preds
        })

        output.to_csv("artifacts/predictions.csv", index=False)

        logging.info("Inference pipeline completed")

    except Exception as e:
        raise customexception(e, sys)