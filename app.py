import sys
from Emotion_State_Prediction.src.pipeline.training_pipeline import run_training
from Emotion_State_Prediction.src.pipeline.inference_pipeline import run_inference
from Emotion_State_Prediction.src.data.load_data import load_test_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py [train|predict]")
        return

    command = sys.argv[1]

    if command == "train":
        run_training()

    elif command == "predict":
        df = load_test_data()
        result = run_inference(df)
        result.to_csv("predictions.csv", index=False)
        print("Predictions saved")

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()