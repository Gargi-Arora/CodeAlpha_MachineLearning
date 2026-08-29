"""
train.py
--------
End-to-end training script for Speech Emotion Recognition.

Usage:
    python train.py /path/to/RAVDESS

Trains three models for comparison:
  1. Baseline: Random Forest on extracted features
  2. Deep model: MLP
  3. Deep model: 1D-CNN + BiLSTM hybrid (best performer, treats feature
     vector as a short sequence)

Saves the best model + label encoder + scaler to ./artifacts/
"""

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Flatten, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from feature_extraction import build_dataframe, build_feature_dataset

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def prepare_data(ravdess_root: str):
    df = build_dataframe(ravdess_root)

    # Stratified split BEFORE augmentation so val/test never contain
    # augmented copies of a train sample (avoids data leakage)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["emotion"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42
    )

    print("Extracting TRAIN features (with augmentation)...")
    X_train, y_train = build_feature_dataset(train_df, augment=True)
    print("Extracting VAL features...")
    X_val, y_val = build_feature_dataset(val_df, augment=False)
    print("Extracting TEST features...")
    X_test, y_test = build_feature_dataset(test_df, augment=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))

    return (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc), le


def build_cnn_lstm(input_dim: int, n_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_and_report(name, y_true, y_pred, label_encoder):
    print(f"\n=== {name} — Classification Report ===")
    target_names = label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=target_names))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name} macro F1: {macro_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, f"confusion_matrix_{name}.png"))
    plt.close()
    return macro_f1


def main(ravdess_root: str):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), le = prepare_data(ravdess_root)
    n_classes = len(le.classes_)

    # ---- Baseline: Random Forest ----
    print("\nTraining baseline Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    evaluate_and_report("RandomForest", y_test, rf_preds, le)
    joblib.dump(rf, os.path.join(ARTIFACT_DIR, "random_forest.pkl"))

    # ---- Deep model: CNN + BiLSTM ----
    print("\nTraining CNN+BiLSTM hybrid...")
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)

    model = build_cnn_lstm(X_train.shape[1], n_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(ARTIFACT_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train_cnn, y_train_cat,
        validation_data=(X_val_cnn, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
