# Disease Prediction from Medical Data

Predicts the possibility of disease (Heart Disease, Diabetes, Breast Cancer)
from structured patient data using classification techniques.

## Datasets
| Dataset | Source | Target |
|---|---|---|
| Heart Disease | UCI Cleveland (mirrored CSV) | Presence of heart disease |
| Diabetes | Pima Indians Diabetes (mirrored CSV) | Diabetes diagnosis |
| Breast Cancer | UCI Wisconsin (built into scikit-learn) | Malignant vs benign tumor |

## Approach
1. **Preprocessing**: median imputation for missing values, feature scaling,
   stratified train/test split, SMOTE oversampling on the training set when
   classes are imbalanced.
2. **Models compared**: Logistic Regression, Random Forest, SVM, XGBoost —
   each tuned with GridSearchCV (5-fold cross-validation).
3. **Evaluation**: accuracy, precision, **recall**, F1, and ROC-AUC are all
   reported — recall is weighted heavily in model selection since missing a
   true disease case (false negative) is more costly than a false alarm in
   a medical screening context.
4. **Best model selection**: highest F1 score per dataset is saved for
   deployment.
5. **Demo**: Streamlit app where a user selects a condition, enters patient
   details, and gets a prediction with probability.

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```
This trains and tunes all 4 models on all 3 datasets, and saves to
`./artifacts/<dataset_name>/`:
- `best_model.pkl`, `scaler.pkl`, `imputer.pkl`, `feature_names.pkl`
- `roc_curves.png`, per-model confusion matrices
- `results_summary.csv` — full metrics table across all datasets/models

## Run the demo
```bash
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `data_loader.py` | Loads and prepares all three datasets |
| `preprocessing.py` | Shared imputation/scaling/SMOTE pipeline |
| `train.py` | Trains, tunes, and evaluates all models on all datasets |
| `app.py` | Streamlit multi-disease prediction demo |
| `requirements.txt` | Python dependencies |

## Notes for report
- Class imbalance handled with SMOTE on the training data only (never on
  test data, to avoid leakage/inflated metrics).
- Recall is emphasized over raw accuracy since false negatives are more
  costly in a screening/diagnostic context.
- Feature importance from Random Forest/XGBoost can be discussed as a
  proxy for which clinical markers matter most per disease.
