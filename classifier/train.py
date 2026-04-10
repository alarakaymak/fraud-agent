"""
Fraud Detection Classifier
--------------------------
Trains an XGBoost classifier on the Kaggle Credit Card Fraud dataset.
Outputs fraud_model.pkl, scaler.pkl, and classifier_metrics.json.

Dataset: mlg-ulb/creditcardfraud (loaded via kagglehub)
"""

import json
import joblib
import pandas as pd
from pathlib import Path
import xgboost as xgb
import kagglehub
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess() -> tuple:
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    print(f"Loaded {len(df):,} transactions — fraud rate: {df['Class'].mean():.4%}")

    df = df.drop_duplicates()

    # Keep a copy of Amount and Time before scaling — used by the agent for human-readable context
    # The LLM never sees V1–V28 (anonymized PCA features); it only sees fraud_score + amount + time_of_day
    df["time_of_day_seconds"] = df["Time"] % 86400  # wrap to 24h cycle

    # Scale Amount and Time with separate scalers (V1–V28 are already PCA'd)
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = time_scaler.fit_transform(df[["Time"]])
    df = df.drop(["Amount", "Time", "time_of_day_seconds"], axis=1)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"]

    return X, y, amount_scaler, time_scaler


def train(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    # Weight to compensate for class imbalance (~577:1 legit:fraud)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # PR-AUC is more informative for imbalanced data
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Calibrate probabilities with isotonic regression.
    # XGBoost with high scale_pos_weight outputs extreme near-0/near-1 scores,
    # making the borderline zone empty. Calibration spreads the distribution so
    # the agent actually fires on ambiguous cases.
    print("\nCalibrating probabilities (isotonic regression)...")
    raw_probs = model.predict_proba(X_test)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_test)
    return model, calibrator


def evaluate(model, calibrator, X_test, y_test) -> dict:
    raw = model.predict_proba(X_test)[:, 1]
    y_prob = calibrator.predict(raw)
    y_pred = (y_prob > 0.5).astype(int)

    print("\n--- Classification Report (threshold=0.5) ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}  ← more important for imbalanced datasets")

    # Show how transactions would be routed by the agent
    # Thresholds chosen from calibrated score distribution:
    # ~94% of transactions score < 0.001 (clearly legit), ~0.4% score > 0.999 (clearly fraud)
    # Everything in between goes to the LLM agent for reasoning
    AUTO_APPROVE_THRESHOLD = 0.001
    AUTO_FLAG_THRESHOLD = 0.999

    auto_approve = y_prob < AUTO_APPROVE_THRESHOLD
    auto_flag = y_prob > AUTO_FLAG_THRESHOLD
    borderline = ~auto_approve & ~auto_flag

    print("\n--- Agent Routing Distribution ---")
    print(f"Auto-approve (<{AUTO_APPROVE_THRESHOLD}):   {auto_approve.mean():.1%} of transactions")
    print(f"Borderline (LLM):          {borderline.mean():.1%} of transactions")
    print(f"Auto-flag (>{AUTO_FLAG_THRESHOLD}):  {auto_flag.mean():.1%} of transactions")
    print(f"Fraud rate in borderline zone: {y_test[borderline].mean():.1%}")

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "auto_approve_threshold": AUTO_APPROVE_THRESHOLD,
        "auto_flag_threshold": AUTO_FLAG_THRESHOLD,
        "auto_approve_pct": round(float(auto_approve.mean()), 4),
        "borderline_pct": round(float(borderline.mean()), 4),
        "auto_flag_pct": round(float(auto_flag.mean()), 4),
        "borderline_fraud_rate": round(float(y_test[borderline].mean()), 4),
    }


def main():
    X, y, amount_scaler, time_scaler = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, calibrator = train(X_train, y_train, X_test, y_test)
    metrics = evaluate(model, calibrator, X_test, y_test)

    # Save artifacts to the classifier directory regardless of where script is run from
    out = Path(__file__).parent
    joblib.dump(model, out / "fraud_model.pkl")
    joblib.dump(calibrator, out / "calibrator.pkl")
    joblib.dump(amount_scaler, out / "amount_scaler.pkl")
    joblib.dump(time_scaler, out / "time_scaler.pkl")
    with open(out / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: fraud_model.pkl, calibrator.pkl, amount_scaler.pkl, time_scaler.pkl, classifier_metrics.json")


if __name__ == "__main__":
    main()
