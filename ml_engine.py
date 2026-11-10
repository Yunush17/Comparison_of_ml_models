"""Reusable training/evaluation code for the ML model comparison project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

BUILTIN_DATASETS = {
    "Iris (classification)": (load_iris, "classification"),
    "Breast Cancer (classification)": (load_breast_cancer, "classification"),
    "Wine (classification)": (load_wine, "classification"),
    "Diabetes (regression)": (load_diabetes, "regression"),
}


def load_builtin(name: str):
    if name not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    loader, task = BUILTIN_DATASETS[name]
    data = loader(as_frame=True)
    frame = data.frame.copy()
    target_col = data.target.name or "target"
    if target_col not in frame.columns:
        frame[target_col] = data.target
    if task == "classification" and hasattr(data, "target_names"):
        frame[target_col] = frame[target_col].map(dict(enumerate(data.target_names)))
    return frame, target_col, task


def build_models(task: str, n_train: int, seed: int = 42):
    k = min(5, n_train)
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=3000, random_state=seed),
            "Decision Tree": DecisionTreeClassifier(random_state=seed),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=k),
            "Naive Bayes": GaussianNB(),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Decision Tree": DecisionTreeRegressor(random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "Support Vector Regression": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=k),
    }


def _make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_models(frame: pd.DataFrame, target: str, task: str,
                    test_size: float = 0.2, seed: int = 42):
    """Returns (sorted metrics df, fitted pipelines, model outputs, holdout size).

    Preprocessing is fit only on training data, preventing test-set leakage.
    """
    if task not in ("classification", "regression"):
        raise ValueError("Task must be 'classification' or 'regression'.")
    if target not in frame.columns:
        raise ValueError("Selected target column is missing.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test size must be between 0.1 and 0.4.")
    df = frame.copy().dropna(subset=[target])
    df = df.drop(columns=[c for c in df.columns if c != target and df[c].isna().all()])
    if task == "regression":
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=[target])
    if len(df) < 20:
        raise ValueError("At least 20 usable rows are required.")
    X, y = df.drop(columns=[target]), df[target]
    if X.shape[1] == 0:
        raise ValueError("The dataset must include at least one feature column.")
    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError("Classification needs at least two target classes.")
        if counts.min() < 2:
            raise ValueError("Each class must have at least two rows for stratified splitting.")
        n_test = int(np.ceil(len(y) * test_size))
        if n_test < len(counts) or len(y) - n_test < len(counts):
            raise ValueError("Increase the data size or test percentage so every class fits in both splits.")
        stratify = y
    else:
        stratify = None
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    # Convert unusual types to string to support CSV-derived mixed data.
    X = X.copy()
    for col in categorical:
        X[col] = X[col].astype("string").astype(object)
        X[col] = X[col].where(pd.notna(X[col]), np.nan)
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _make_encoder())]), categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify)
    model_results = []
    fitted, outputs, errors = {}, {}, {}
    for name, estimator in build_models(task, len(X_train), seed).items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            if task == "classification":
                metrics = {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Error Rate": 1 - accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
                }
            else:
                mse = mean_squared_error(y_test, pred)
                metrics = {"Model": name, "MAE": mean_absolute_error(y_test, pred),
                           "MSE": mse, "RMSE": float(np.sqrt(mse)),
                           "R2 Score": r2_score(y_test, pred)}
            model_results.append(metrics)
            fitted[name] = pipe
            outputs[name] = {"actual": np.asarray(y_test), "predicted": np.asarray(pred)}
        except Exception as exc:
            errors[name] = str(exc)
    if not model_results:
        raise ValueError("No models could be fitted. " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    result_df = pd.DataFrame(model_results)
    sort_by = "Accuracy" if task == "classification" else "RMSE"
    result_df = result_df.sort_values(sort_by, ascending=task == "regression").reset_index(drop=True)
    return result_df, fitted, outputs, len(X_test), errors
