import pickle
import pandas as pd
import numpy as np

from Emotion_State_Prediction.src.pipeline.decision_engine import decision_engine


def run_inference(input_df):

    input_df = input_df.copy()
    input_df["journal_text"] = input_df["journal_text"].fillna("")
    input_df["stress_level"] = input_df["stress_level"].fillna(5)
    input_df["energy_level"] = input_df["energy_level"].fillna(5)

    with open("artifacts/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("artifacts/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open("artifacts/intensity_model.pkl", "rb") as f:
        intensity_model = pickle.load(f)

    X = vectorizer.transform(input_df["journal_text"])

    state_pred_encoded = model.predict(X).astype(int)
    state_pred = le.inverse_transform(state_pred_encoded)

    intensity_pred = intensity_model.predict(X)
    intensity_pred = np.round(intensity_pred).astype(int)
    intensity_pred = np.clip(intensity_pred, 1, 5)

    try:
        probs = model.predict_proba(X)
        confidence = np.max(probs, axis=1)
    except:
        confidence = np.ones(len(X)) * 0.7

    uncertain_flag = [1 if c < 0.6 else 0 for c in confidence]

    actions = []
    timings = []

    for i in range(len(input_df)):

        text = input_df["journal_text"].iloc[i]
        if len(text.strip()) < 5:
            actions.append("pause")
            timings.append("now")
            continue

        action, timing = decision_engine(
            state_pred[i],
            intensity_pred[i],
            input_df["stress_level"].iloc[i],
            input_df["energy_level"].iloc[i],
            input_df["time_of_day"].iloc[i],
        )

        actions.append(action)
        timings.append(timing)

    result = pd.DataFrame({
        "id": input_df["id"],
        "predicted_state": state_pred,
        "predicted_intensity": intensity_pred,
        "confidence": confidence,
        "uncertain_flag": uncertain_flag,
        "what_to_do": actions,
        "when_to_do": timings
    })

    return result