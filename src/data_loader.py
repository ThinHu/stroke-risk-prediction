import os
import pandas as pd
import kagglehub

def load_stroke_data(download_path="fedesoriano/stroke-prediction-dataset"):
    """Downloads the dataset from Kaggle and loads it into a DataFrame."""
    path = kagglehub.dataset_download(download_path)
    file_path = os.path.join(path, "healthcare-dataset-stroke-data.csv")
    return pd.read_csv(file_path)