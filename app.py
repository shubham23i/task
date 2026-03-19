from Emotion_State_Prediction.src.pipeline.training_pipeline import run_training
from Emotion_State_Prediction.src.pipeline.inference_pipeline import run_inference
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py [train|predict]")
        return

    if sys.argv[1] == "train":
        run_training()

    elif sys.argv[1] == "predict":
        run_inference()

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()