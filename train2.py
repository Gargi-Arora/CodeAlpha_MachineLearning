"""
train.py
--------
Trains Logistic Regression, Random Forest, SVM, and XGBoost on each of
the three datasets (heart disease, diabetes, breast cancer), tunes each
with GridSearchCV, evaluates with medically-relevant metrics (recall,
F1, ROC-AUC — not just accuracy), and saves the best model per dataset.

Usage:
    python train.py

Saves to ./artifacts/<dataset_name>/:
    best_model.pkl, scaler.pkl, imputer.pkl, feature_names.pkl,
    roc_curve.png, confusion matrices, results_summary.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from data_loader import load_all_datasets
from preprocessing import preprocess

ARTIFACT_DIR = "artifacts"

MODEL_GRID = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": [200, 400], "max_depth": [None, 6, 10]},
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced"),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
}


def train_and_evaluate_dataset(name, X, y, feature_names):
    out_dir = os.path.join(ARTIFACT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, imputer = preprocess(X, y)

    results = []
    plt.figure(figsize=(7, 6))

    best_model, best_model_name, best_f1 = None, None, -1

    for model_name, (estimator, param_grid) in MODEL_GRID.items():
        print(f"\n[{name}] Tuning {model_name}...")
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print(f"{model_name}: acc={acc:.3f} prec={prec:.3f} recall={rec:.3f} f1={f1:.3f} auc={auc:.3f}")
        print(classification_report(y_test, preds))

        results.append({
            "dataset": name, "model": model_name, "best_params": grid.best_params_,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc,
        })

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} — {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"))
        plt.close()

        # Selection priority: recall matters most in medical screening,
        # so we pick the model with the best F1 (balances recall + precision)
        if f1 > best_f1:
            best_f1, best_model, best_model_name = f1, model, model_name

    plt.figure(1)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"))
    plt.close()

    joblib.dump(best_model, os.path.join(out_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_names.pkl"))

    print(f"\n[{name}] Best model: {best_model_name} (F1={best_f1:.3f}) saved to {out_dir}/best_model.pkl")
    return results


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    datasets = load_all_datasets()

    all_results = []
    for name, (X, y, feature_names) in datasets.items():
        results = train_and_evaluate_dataset(name, X, y, feature_names)
        all_results.extend(results)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(ARTIFACT_DIR, "results_summary.csv"), index=False)
    print("\n\n=== FINAL RESULTS SUMMARY ===")
    print(summary_df[["dataset", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()
