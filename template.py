import os
from pathlib import Path

project_name = "Emotion State Prediction"


list_of_files = [
    f"{project_name}/data/raw/test.csv",
    f"{project_name}/data/raw/train.csv",
    f"{project_name}/data/processed/.gitkeep",

    f"{project_name}/notebooks/EDA.ipynb",

    f"{project_name}/src/__init__.py",

    f"{project_name}/src/data/__init__.py",
    f"{project_name}/src/data/load_data.py",
    f"{project_name}/src/data/preprocess.py",

    f"{project_name}/src/features/__init__.py",
    f"{project_name}/src/features/build_features.py",

    f"{project_name}/src/models/__init__.py",
    f"{project_name}/src/models/train.py",
    f"{project_name}/src/models/predict.py",

    f"{project_name}/src/pipeline/__init__.py",
    f"{project_name}/src/pipeline/training_pipeline.py",
    f"{project_name}/src/pipeline/inference_pipeline.py",

    f"{project_name}/artifacts/.gitkeep",

    f"{project_name}/config/config.yaml",

    "app.py",

    f"{project_name}/requirements.txt",
    f"{project_name}/README.md",
    f"{project_name}/.gitignore",
    f"{project_name}/setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"{filename} already exists")


