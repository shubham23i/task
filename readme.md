# Emotion State Prediction System

## Overview

This project presents a system designed to understand a user’s emotional state from short and often noisy journal reflections. Unlike traditional classification tasks, the goal is not just to predict emotions but to guide users toward a better mental state. The system combines text understanding, contextual signals, and decision-making logic to provide meaningful recommendations.

## Approach

The core of the system is built around processing the `journal_text` field using TF-IDF vectorization. This approach was chosen because it is lightweight, interpretable, and performs well on short and unstructured text. Alongside text, contextual features such as stress level, energy level, and time of day are incorporated indirectly through the decision layer.

Multiple machine learning models were trained, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and CatBoost. Hyperparameter tuning was performed using GridSearchCV, and the best-performing model was selected based on validation accuracy.

## Intensity Prediction

In addition to predicting emotional state, the system also predicts emotional intensity on a scale of 1 to 5. This was treated as a regression problem to better capture the continuous nature of emotional strength. A RandomForestRegressor was used for this task, and predictions were clipped to ensure they remain within the valid range.

## Decision Engine

A rule-based decision engine was implemented to convert predictions into actionable guidance. The system considers predicted emotional state, intensity, stress, energy, and time of day to decide what action the user should take and when they should take it. For example, a highly overwhelmed user is guided toward immediate breathing exercises, while a calm user is encouraged to engage in deep work.

## Uncertainty Handling

To make the system more reliable, uncertainty is explicitly modeled using prediction probabilities. A confidence score is extracted from the model, and predictions with confidence below a threshold are flagged as uncertain. This ensures the system acknowledges when it is not confident rather than making misleading recommendations.

## Features

The system relies primarily on textual input, which provides the strongest signal for emotional understanding. Metadata such as stress level and energy level plays a supporting role in decision-making rather than direct prediction.

## How to Run

The system can be executed using two commands. First, the training pipeline is run to build and save the models. Then, the inference pipeline is used to generate predictions on test data.

## Output

The final output is a CSV file containing predicted emotional state, predicted intensity, confidence score, uncertainty flag, and recommended actions with timing.

## Future Improvements

Future work could include using transformer-based models for better text understanding, implementing multi-label classification to handle mixed emotions, improving uncertainty calibration, and replacing the rule-based decision system with a learned policy.
