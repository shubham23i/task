import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import mlflow
import dagshub

from Emotion_State_Prediction.src.exception import customexception
from Emotion_State_Prediction.src.logger import logging
from Emotion_State_Prediction.src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

        dagshub.init(
            repo_owner="shubham23i",
            repo_name="Emotion_State_Prediction",
            mlflow=True,
        )

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
                "CatBoost":CatBoostClassifier(verbose=0,train_dir="artifacts/catboost_info"),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5, 10],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100],
                },
                "XGBoost": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100],
                },
            }

            best_model = None
            best_score = 0
            best_model_name = ""

            logging.info("Starting model training with GridSearchCV")

            for model_name, model in models.items():

                logging.info(f"Training {model_name}")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    cv=3,
                    n_jobs=-1,
                    scoring="accuracy",
                )

                grid.fit(X_train, y_train)

                trained_model = grid.best_estimator_

                y_pred = trained_model.predict(X_test)
                score = accuracy_score(y_test, y_pred)

                logging.info(f"{model_name} Accuracy: {score}")

                if score > best_score:
                    best_score = score
                    best_model = trained_model
                    best_model_name = model_name

            if best_model is None:
                raise customexception("No model was trained properly", sys)

            logging.info(f"Best Model: {best_model_name} | Score: {best_score}")

            print("\n Best Model:", best_model_name)
            print(" Best Accuracy:", best_score)

            mlflow.set_experiment("Emotion_State_Prediction")

            with mlflow.start_run():

                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_params(best_model.get_params())
                mlflow.log_metric("accuracy", best_score)

                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name=best_model_name
                )

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Model saved successfully")

            return best_score

        except Exception as e:
            raise customexception(e, sys)