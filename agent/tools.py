"""
Agent tools — 5 tools the LLM can call to investigate a transaction.
Uses DynamoDB for user history when DYNAMODB_HISTORY_TABLE is set,
falls back to mock data for local dev without AWS.
"""

import json
import os
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key
from langchain_core.tools import tool


class _DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, indent=2, cls=_DecimalEncoder)

# ---------------------------------------------------------------------------
# Mock data — used when DYNAMODB_HISTORY_TABLE is not set
# ---------------------------------------------------------------------------

import time as _time

MOCK_HISTORY = {
    "user_001": [
        {"amount": 23.50,  "merchant": "Whole Foods",       "time": "08:30 AM", "city": "Nashville, TN",    "status": "approved", "timestamp": int(_time.time()) - 86400 * 2},
        {"amount": 156.20, "merchant": "Shell Gas Station",  "time": "05:15 PM", "city": "Nashville, TN",    "status": "approved", "timestamp": int(_time.time()) - 86400 * 3},
        {"amount": 45.00,  "merchant": "Chipotle",           "time": "07:45 PM", "city": "Nashville, TN",    "status": "approved", "timestamp": int(_time.time()) - 86400 * 4},
        {"amount": 12.99,  "merchant": "Netflix",            "time": "09:00 AM", "city": "Nashville, TN",    "status": "approved", "timestamp": int(_time.time()) - 86400 * 5},
        {"amount": 89.99,  "merchant": "Gap",                "time": "02:30 PM", "city": "Nashville, TN",    "status": "approved", "timestamp": int(_time.time()) - 86400 * 6},
    ],
    "user_002": [
        {"amount": 4200.00, "merchant": "Best Buy",          "time": "11:00 AM", "city": "San Francisco, CA", "status": "approved", "timestamp": int(_time.time()) - 3600 * 2},
        {"amount": 3800.00, "merchant": "Apple Store",       "time": "03:20 PM", "city": "San Francisco, CA", "status": "approved", "timestamp": int(_time.time()) - 3600 * 5},
        {"amount": 2900.00, "merchant": "Samsung",           "time": "01:45 PM", "city": "San Francisco, CA", "status": "approved", "timestamp": int(_time.time()) - 86400},
        {"amount": 500.00,  "merchant": "Amazon",            "time": "10:00 AM", "city": "San Francisco, CA", "status": "approved", "timestamp": int(_time.time()) - 86400 * 2},
        {"amount": 1200.00, "merchant": "Dell",              "time": "04:00 PM", "city": "San Francisco, CA", "status": "approved", "timestamp": int(_time.time()) - 86400 * 3},
    ],
    "user_003": [
        # Simulates burst activity — 4 transactions within 10 minutes, same city
        {"amount": 15.00,  "merchant": "Starbucks",          "time": "07:00 AM", "city": "Chicago, IL",      "status": "approved", "timestamp": int(_time.time()) - 300},
        {"amount": 8.50,   "merchant": "McDonald's",         "time": "07:02 AM", "city": "Chicago, IL",      "status": "approved", "timestamp": int(_time.time()) - 480},
        {"amount": 22.00,  "merchant": "CVS Pharmacy",       "time": "07:04 AM", "city": "Chicago, IL",      "status": "approved", "timestamp": int(_time.time()) - 540},
        {"amount": 65.00,  "merchant": "Uber",               "time": "07:06 AM", "city": "Chicago, IL",      "status": "approved", "timestamp": int(_time.time()) - 590},
        {"amount": 30.00,  "merchant": "Trader Joe's",       "time": "07:08 AM", "city": "Chicago, IL",      "status": "approved", "timestamp": int(_time.time()) - 610},
    ],
}

# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

_dynamodb = None

def _get_table(table_name: str):
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        )
    return _dynamodb.Table(table_name)


def get_user_history(user_id: str) -> list:
    """
    Retrieve last 5 transactions for a user.
    Uses DynamoDB if DYNAMODB_HISTORY_TABLE env var is set, else mock data.
    """
    table_name = os.environ.get("DYNAMODB_HISTORY_TABLE")
    if not table_name:
        return MOCK_HISTORY.get(user_id, [])

    table = _get_table(table_name)
    response = table.query(
        KeyConditionExpression=Key("user_id").eq(user_id),
        ScanIndexForward=False,  # newest first
        Limit=5,
    )
    return response.get("Items", [])


def log_decision(transaction_id: str, user_id: str, fraud_score: float,
                 decision: str, explanation: str, tools_called: list) -> None:
    """
    Write agent decision to DynamoDB for observability and metrics.
    No-ops if DYNAMODB_DECISIONS_TABLE is not set.
    """
    table_name = os.environ.get("DYNAMODB_DECISIONS_TABLE")
    if not table_name:
        return

    import time
    table = _get_table(table_name)
    table.put_item(Item={
        "transaction_id": transaction_id,
        "timestamp": int(time.time()),
        "user_id": user_id,
        "fraud_score": Decimal(str(round(fraud_score, 6))),
        "decision": decision,
        "explanation": explanation,
        "tools_called": tools_called,
    })


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def lookup_user_history(user_id: str) -> str:
    """Look up the last 5 transactions for a user to understand their normal spending behaviour."""
    history = get_user_history(user_id)
    if not history:
        return f"No transaction history found for user {user_id}."
    return _dumps(history)


@tool
def analyze_spending_pattern(user_id: str, current_amount: float) -> str:
    """
    Analyse whether the current transaction amount is unusual compared to
    the user's historical spending. Returns average, max, and a flag if
    the amount is more than 3x the user's average.
    """
    history = get_user_history(user_id)
    if not history:
        return json.dumps({"error": "No history available to analyse."})

    amounts = [float(t["amount"]) for t in history]
    avg = sum(amounts) / len(amounts)
    ratio = current_amount / avg if avg > 0 else 0

    return _dumps({
        "average_transaction": round(avg, 2),
        "max_transaction": float(max(amounts)),
        "current_amount": current_amount,
        "times_above_average": round(ratio, 1),
        "is_unusual": ratio > 3.0,
    })


@tool
def check_velocity(user_id: str, window_minutes: int = 60) -> str:
    """
    Detect high-frequency transaction patterns for this user.
    Checks both a time-window count and the density between recent transactions.
    A burst of 3+ transactions within 15 minutes is a strong card-testing signal.
    """
    table_name = os.environ.get("DYNAMODB_HISTORY_TABLE")
    cutoff = int(_time.time()) - (window_minutes * 60)

    if table_name:
        table = _get_table(table_name)
        # Count within the rolling window
        windowed = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id) & Key("timestamp").gt(cutoff),
        )
        recent_window = windowed.get("Items", [])
        # Fetch the most recent 10 transactions for burst-density check
        latest = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=10,
        )
        recent_all = latest.get("Items", [])
        # If DynamoDB returned nothing, fall back to mock data for burst detection
        if not recent_all:
            recent_all = sorted(MOCK_HISTORY.get(user_id, []),
                                key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
    else:
        recent_window = [t for t in MOCK_HISTORY.get(user_id, []) if t.get("timestamp", 0) > cutoff]
        recent_all = sorted(MOCK_HISTORY.get(user_id, []),
                            key=lambda x: x.get("timestamp", 0), reverse=True)[:10]

    window_count = len(recent_window)
    is_high_velocity = window_count >= 3

    # Burst detection: 3+ transactions within any 15-minute span among the most recent items
    burst_signal = ""
    if not is_high_velocity and len(recent_all) >= 3:
        timestamps = sorted([float(t.get("timestamp", 0)) for t in recent_all], reverse=True)
        for i in range(len(timestamps) - 2):
            span_sec = timestamps[i] - timestamps[i + 2]
            if span_sec <= 900:  # 15 minutes
                burst_count = sum(1 for ts in timestamps if ts >= timestamps[i + 2])
                span_min = round(span_sec / 60, 1)
                is_high_velocity = True
                burst_signal = (
                    f"{burst_count} transactions within {span_min} minutes "
                    f"— BURST PATTERN, possible card testing"
                )
                break

    if burst_signal:
        # Burst found in recent history — return only the burst info so the
        # specialist isn't confused by a zero window count alongside HIGH risk.
        return _dumps({
            "burst_pattern_detected": True,
            "risk_indicator": "HIGH — burst of rapid transactions detected (possible card testing)",
            "signal": burst_signal,
        })
    elif is_high_velocity:
        signal = f"{window_count} transaction(s) in the last {window_minutes} minutes — HIGH VELOCITY, possible card testing"
    else:
        signal = f"{window_count} transaction(s) in the last {window_minutes} minutes — normal activity"

    return _dumps({
        "transactions_in_window": window_count,
        "window_minutes": window_minutes,
        "burst_pattern_detected": is_high_velocity,
        "signal": signal,
    })


@tool
def check_temporal_pattern(user_id: str, current_time_of_day: str) -> str:
    """
    Compare the current transaction time against this user's historical
    transaction times. A transaction at 2AM when the user always transacts
    between 8AM-9PM is a strong anomaly signal.
    """
    import re

    def parse_hour(t: str) -> float | None:
        m = re.match(r"(\d+):(\d+)\s*(AM|PM)", str(t).strip().upper())
        if not m:
            return None
        h, mi, period = int(m.group(1)), int(m.group(2)), m.group(3)
        if period == "PM" and h != 12:
            h += 12
        if period == "AM" and h == 12:
            h = 0
        return h + mi / 60

    history = get_user_history(user_id)
    current_hour = parse_hour(current_time_of_day)

    if not history or current_hour is None:
        return json.dumps({"error": "Insufficient data for temporal analysis."})

    hours = [parse_hour(t.get("time", "")) for t in history]
    hours = [h for h in hours if h is not None]

    if not hours:
        return json.dumps({"error": "No valid timestamps in history."})

    avg_hour = sum(hours) / len(hours)
    min_hour = min(hours)
    max_hour = max(hours)

    # Hours outside the user's normal window
    is_unusual_time = current_hour < min_hour - 1 or current_hour > max_hour + 1
    typical_window = f"{int(min_hour):02d}:00 – {int(max_hour):02d}:59"

    # Compute risk deterministically — midnight–6 AM is HIGH (fraud hours),
    # evening outside window is MEDIUM, within window is LOW.
    if is_unusual_time:
        if current_hour < 6:
            computed_risk = "HIGH"
            time_context  = "FRAUD HOURS (midnight–6 AM)"
        else:
            computed_risk = "MEDIUM"
            time_context  = "evening hours (7 PM–midnight) — unusual but not fraud-specific"
    else:
        computed_risk = "LOW"
        time_context  = "within normal hours"

    return _dumps({
        "current_hour": round(current_hour, 2),
        "user_typical_window": typical_window,
        "average_transaction_hour": round(avg_hour, 2),
        "is_unusual_time": is_unusual_time,
        "computed_risk_level": computed_risk,
        "signal": f"Transaction at {current_time_of_day}; user typically transacts {typical_window} — {time_context}",
    })


@tool
def check_location(user_id: str, current_city: str) -> str:
    """
    Compare the current transaction city against the user's last known location
    and the time elapsed. Detects impossible travel — e.g. Nashville at 3PM and
    Los Angeles at 2AM the same night is physically impossible.
    """
    history = get_user_history(user_id)
    if not history:
        return json.dumps({"error": "No history available for location check."})

    # Find the most recent transaction with a city
    last_txn = None
    for t in sorted(history, key=lambda x: float(x.get("timestamp", 0)), reverse=True):
        if t.get("city"):
            last_txn = t
            break

    if not last_txn:
        return json.dumps({"signal": "No prior location data available."})

    last_city = last_txn.get("city", "Unknown")
    last_ts = float(last_txn.get("timestamp", 0))
    hours_elapsed = (_time.time() - last_ts) / 3600

    same_city = last_city.split(",")[0].strip().lower() == current_city.split(",")[0].strip().lower()

    return _dumps({
        "last_known_city": last_city,
        "current_city": current_city,
        "hours_since_last_transaction": round(hours_elapsed, 1),
        "same_city": same_city,
        "signal": (
            f"Last transaction in {last_city} ({round(hours_elapsed, 1)}h ago). "
            f"Current transaction in {current_city}. "
            + ("Same city — no location anomaly." if same_city
               else f"DIFFERENT CITY — assess whether travel from {last_city} to {current_city} in {round(hours_elapsed, 1)} hours is physically possible.")
        ),
    })


@tool
def block_transaction(reason: str) -> str:
    """
    BLOCK this transaction as confirmed fraud. Use when you have strong converging evidence:
    high fraud score + unusual amount + unusual time + high velocity or cardholder denial.
    """
    return json.dumps({"decision": "BLOCK", "reason": reason})


@tool
def approve_transaction(reason: str) -> str:
    """
    APPROVE this transaction as legitimate. Use when evidence strongly supports it:
    normal amount, normal time, matches spending history, cardholder confirmed.
    """
    return json.dumps({"decision": "APPROVE", "reason": reason})


@tool
def review_transaction(reason: str) -> str:
    """
    Send to human REVIEW queue. Use when signals are mixed and you cannot reach
    a confident decision — some fraud indicators present but not conclusive.
    """
    return json.dumps({"decision": "REVIEW", "reason": reason})


@tool
def request_clarification(question: str) -> str:
    """
    Ask the cardholder to verify the transaction. Use when a simple confirmation
    would resolve ambiguity before making a final decision.
    """
    return json.dumps({"decision": "CLARIFICATION_NEEDED", "question": question})


ALL_TOOLS = [
    lookup_user_history,
    analyze_spending_pattern,
    check_velocity,
    check_temporal_pattern,
    check_location,
    block_transaction,
    approve_transaction,
    review_transaction,
    request_clarification,
]
