"""
Classifier inference — loads trained artifacts and scores a transaction.
"""

import joblib
import pandas as pd
from pathlib import Path

CLASSIFIER_DIR = Path(__file__).parent.parent / "classifier"

_model = None
_calibrator = None
_amount_scaler = None
_time_scaler = None


def _load():
    global _model, _calibrator, _amount_scaler, _time_scaler
    if _model is None:
        _model = joblib.load(CLASSIFIER_DIR / "fraud_model.pkl")
        _calibrator = joblib.load(CLASSIFIER_DIR / "calibrator.pkl")
        _amount_scaler = joblib.load(CLASSIFIER_DIR / "amount_scaler.pkl")
        _time_scaler = joblib.load(CLASSIFIER_DIR / "time_scaler.pkl")


def scale_features(amount: float, time_seconds: float) -> tuple:
    """Return (amount_scaled, time_scaled) using the trained scalers."""
    _load()
    amount_scaled = float(_amount_scaler.transform(pd.DataFrame([{"Amount": amount}]))[0][0])
    time_scaled = float(_time_scaler.transform(pd.DataFrame([{"Time": time_seconds}]))[0][0])
    return amount_scaled, time_scaled


def score(feature_row: dict) -> float:
    """
    Score a transaction given a dict of features matching the training schema:
    V1–V28, Amount_scaled, Time_scaled.
    Returns a calibrated fraud probability in [0, 1].
    """
    _load()
    df = pd.DataFrame([feature_row])
    raw = _model.predict_proba(df)[:, 1]
    return float(_calibrator.predict(raw)[0])


AUTO_APPROVE_THRESHOLD = 0.001


def route(fraud_score: float) -> str:
    """Return routing decision based on calibrated fraud score."""
    if fraud_score < AUTO_APPROVE_THRESHOLD:
        return "auto_approve"
    return "agent"
