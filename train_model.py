"""
Titanic Survival Prediction - Model Training Script
Trains 5 ML models + 2 DL models, compares them, saves best ML pipeline.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier

# ---------------- 1. LOAD DATA ----------------
df = pd.read_csv("Titanic-Dataset.csv")

# ---------------- 2. CLEANING ----------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])

# ---------------- 3. FEATURES / TARGET ----------------
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

X = df[numeric_features + categorical_features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

# ---------------- 4. PREPROCESSING ----------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------- 5. DEFINE ML MODELS ----------------
ml_models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42, max_depth=5))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42
        ))
    ]),
}

# ---------------- 6. TRAIN + EVALUATE ML ----------------
def evaluate(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }

ml_results = []
trained_ml = {}

for name, model in ml_models.items():
    print(f"\n▶ Training {name} ...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    ml_results.append(evaluate(name, y_test, pred, prob))
    trained_ml[name] = model

ml_results_df = pd.DataFrame(ml_results)

# ---------------- 7. DEEP LEARNING (optional if TF installed) ----------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

    dl_preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]),
         categorical_features)
    ])
    X_train_dl = dl_preprocessor.fit_transform(X_train).toarray()
    X_test_dl = dl_preprocessor.transform(X_test).toarray()

    # ANN
    ann_model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_dl.shape[1],)),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann_model.fit(X_train_dl, y_train, validation_split=0.20,
                  epochs=50, batch_size=32, verbose=0)

    ann_prob = ann_model.predict(X_test_dl).ravel()
    ann_pred = (ann_prob >= 0.5).astype(int)
    ml_results.append(evaluate("ANN", y_test, ann_pred, ann_prob))

    # 1D CNN
    X_train_cnn = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], 1)
    X_test_cnn = X_test_dl.reshape(X_test_dl.shape[0], X_test_dl.shape[1], 1)

    cnn_model = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.30),
        Dense(1, activation="sigmoid"),
    ])
    cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    cnn_model.fit(X_train_cnn, y_train, validation_split=0.20,
                  epochs=50, batch_size=32, verbose=0)

    cnn_prob = cnn_model.predict(X_test_cnn).ravel()
    cnn_pred = (cnn_prob >= 0.5).astype(int)
    ml_results.append(evaluate("1D CNN", y_test, cnn_pred, cnn_prob))
except Exception as e:
    print("DL skipped:", e)

# ---------------- 8. COMPARE & SELECT BEST ----------------
all_results = pd.DataFrame(ml_results).sort_values(by="F1 Score", ascending=False)
print("\n===== MODEL COMPARISON =====")
print(all_results.to_string(index=False))

best_model_name = all_results.iloc[0]["Model"]
print(f"\n🏆 Best model based on F1 Score: {best_model_name}")

# ---------------- 9. SAVE BEST ML PIPELINE ----------------
if best_model_name in trained_ml:
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_ml[best_model_name], "models/titanic_best_model.pkl")
    print("✅ Saved models/titanic_best_model.pkl")
else:
    # If best is a DL model, fall back to best ML model
    best_ml_name = ml_results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    joblib.dump(trained_ml[best_ml_name], "models/titanic_best_model.pkl")
    print(f"✅ Saved best ML fallback: {best_ml_name}")