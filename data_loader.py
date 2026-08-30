"""
data_loader.py
--------------
Loads the three medical datasets used in this project:
  1. Heart Disease (UCI Cleveland, via a mirrored CSV on GitHub)
  2. Diabetes (Pima Indians, via a mirrored CSV on GitHub)
  3. Breast Cancer (Wisconsin, built into scikit-learn)

Each loader returns (X, y, feature_names) where y is binary
(1 = disease present, 0 = not present).

NOTE: The heart/diabetes CSVs are pulled from third-party GitHub mirrors
of the original UCI datasets, since UCI's own URLs change over time.
If a link is ever dead, download the dataset manually from
https://archive.ics.uci.edu/ (Heart Disease / Pima Diabetes) or Kaggle,
save it as heart.csv / diabetes.csv locally, and the functions below
will still work by pointing `path` at your local file instead.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

HEART_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
DIABETES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"


def load_heart(path: str = HEART_URL):
    df = pd.read_csv(path)
    df = df.dropna()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return X, y, list(X.columns)


def load_diabetes_data(path: str = DIABETES_URL):
    df = pd.read_csv(path)
    df = df.dropna()
    # Pima dataset uses 0s as missing values for some clinical columns —
    # replace with NaN so the imputer in preprocessing.py can handle them
    cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_with_invalid_zero:
        if c in df.columns:
            df[c] = df[c].astype(float).replace(0, np.nan)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)
    return X, y, list(X.columns)


def load_breast_cancer_data():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"].astype(int)
    # sklearn encodes 0 = malignant, 1 = benign; flip so 1 = disease (malignant)
    # to keep "1 = disease present" consistent across all three datasets
    y = 1 - y
    return X, y, list(X.columns)


def load_all_datasets():
    """Returns a dict: {dataset_name: (X, y, feature_names)}"""
    return {
        "heart_disease": load_heart(),
        "diabetes": load_diabetes_data(),
        "breast_cancer": load_breast_cancer_data(),
    }


if __name__ == "__main__":
    datasets = load_all_datasets()
    for name, (X, y, features) in datasets.items():
        print(f"\n{name}: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class balance:\n{y.value_counts(normalize=True)}")
