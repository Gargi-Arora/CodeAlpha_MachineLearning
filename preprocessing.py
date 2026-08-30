"""
preprocessing.py
----------------
Shared preprocessing pipeline used for all three datasets:
  1. Median imputation for missing values
  2. Stratified train/test split
  3. Feature scaling (StandardScaler)
  4. SMOTE oversampling on the TRAIN split only (to fix class imbalance)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess(X, y, test_size: float = 0.2, random_state: int = 42, use_smote: bool = True):
    """
    Returns: X_train, X_test, y_train, y_test, scaler, imputer
    (scaler/imputer are returned so the same transforms can be reused
    at inference time in the Streamlit app)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_smote:
        class_counts = np.bincount(y_train)
        imbalance_ratio = class_counts.min() / class_counts.max()
        if imbalance_ratio < 0.8:  # only bother if meaningfully imbalanced
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler, imputer
