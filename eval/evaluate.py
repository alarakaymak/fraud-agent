"""
Evaluation script — runs 40 test cases through the /analyze endpoint
and measures accuracy, routing distribution, and latency.

Usage:
    python evaluate.py [--url http://localhost:8000]
"""

import argparse
import json
import time
import statistics
import requests
from datetime import datetime

# ---------------------------------------------------------------------------
# Test suite — 40 cases covering all 4 decision paths
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ── AUTO-APPROVE (score < 0.001, no LLM call) ──
    {"label": "auto_approve", "expected_set": {"APPROVE", "APPROVED"}, "input": {
        "user_id": "user_001", "amount": 23.50, "merchant_category": "Grocery",
        "time_of_day": "08:30 AM", "city": "Nashville, TN", "fraud_score_override": 0.0001}},
    {"label": "auto_approve", "expected_set": {"APPROVE", "APPROVED"}, "input": {
        "user_id": "user_001", "amount": 12.99, "merchant_category": "Streaming",
        "time_of_day": "09:00 AM", "city": "Nashville, TN", "fraud_score_override": 0.00005}},
    {"label": "auto_approve", "expected_set": {"APPROVE", "APPROVED"}, "input": {
        "user_id": "user_002", "amount": 500.00, "merchant_category": "Electronics",
        "time_of_day": "10:00 AM", "city": "San Francisco, CA", "fraud_score_override": 0.0002}},
    {"label": "auto_approve", "expected_set": {"APPROVE", "APPROVED"}, "input": {
        "user_id": "user_001", "amount": 45.00, "merchant_category": "Restaurant",
        "time_of_day": "07:45 PM", "city": "Nashville, TN", "fraud_score_override": 0.00008}},
    {"label": "auto_approve", "expected_set": {"APPROVE", "APPROVED"}, "input": {
        "user_id": "user_002", "amount": 1200.00, "merchant_category": "Electronics",
        "time_of_day": "04:00 PM", "city": "San Francisco, CA", "fraud_score_override": 0.0003}},

    # ── LEGITIMATE HIGH-SPENDER (should APPROVE) ──
    {"label": "normal_big_spender", "expected": "APPROVE", "input": {
        "user_id": "user_002", "amount": 3500.00, "merchant_category": "Electronics",
        "time_of_day": "02:00 PM", "city": "San Francisco, CA", "fraud_score_override": 0.45}},
    {"label": "normal_big_spender", "expected": "APPROVE", "input": {
        "user_id": "user_002", "amount": 4000.00, "merchant_category": "Electronics",
        "time_of_day": "11:00 AM", "city": "San Francisco, CA", "fraud_score_override": 0.40}},
    {"label": "normal_big_spender", "expected": "APPROVE", "input": {
        "user_id": "user_002", "amount": 2800.00, "merchant_category": "Electronics",
        "time_of_day": "03:00 PM", "city": "San Francisco, CA", "fraud_score_override": 0.38}},

    # ── HIGH FRAUD (should BLOCK) ──
    {"label": "block_fraud", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 1250.00, "merchant_category": "Electronics",
        "time_of_day": "02:14 AM", "city": "Los Angeles, CA", "fraud_score_override": 0.73}},
    {"label": "block_fraud", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 2000.00, "merchant_category": "Electronics",
        "time_of_day": "03:00 AM", "city": "Miami, FL", "fraud_score_override": 0.85}},
    {"label": "block_fraud", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 1800.00, "merchant_category": "Jewelry",
        "time_of_day": "01:45 AM", "city": "New York, NY", "fraud_score_override": 0.92}},
    {"label": "block_fraud", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 999.00, "merchant_category": "Gift Cards",
        "time_of_day": "04:30 AM", "city": "Seattle, WA", "fraud_score_override": 0.88}},
    {"label": "block_fraud", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 3000.00, "merchant_category": "Electronics",
        "time_of_day": "02:00 AM", "city": "Chicago, IL", "fraud_score_override": 0.95}},

    # ── CARD TESTING / VELOCITY (should BLOCK or REVIEW) ──
    {"label": "velocity_fraud", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_003", "amount": 12.00, "merchant_category": "Online Retail",
        "time_of_day": "07:09 AM", "city": "Chicago, IL", "fraud_score_override": 0.55}},
    {"label": "velocity_fraud", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_003", "amount": 5.00, "merchant_category": "Online Retail",
        "time_of_day": "07:10 AM", "city": "Chicago, IL", "fraud_score_override": 0.58}},
    {"label": "velocity_fraud", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_003", "amount": 9.99, "merchant_category": "Gas Station",
        "time_of_day": "07:08 AM", "city": "Chicago, IL", "fraud_score_override": 0.60}},

    # ── BORDERLINE (agent must decide — expect CLARIFICATION_NEEDED or REVIEW) ──
    {"label": "borderline", "expected_set": {"CLARIFICATION_NEEDED", "REVIEW", "BLOCK"}, "input": {
        "user_id": "user_001", "amount": 800.00, "merchant_category": "Electronics",
        "time_of_day": "11:00 PM", "city": "Nashville, TN", "fraud_score_override": 0.38}},
    {"label": "borderline", "expected_set": {"CLARIFICATION_NEEDED", "REVIEW", "APPROVE"}, "input": {
        "user_id": "user_001", "amount": 500.00, "merchant_category": "Jewelry",
        "time_of_day": "10:30 PM", "city": "Nashville, TN", "fraud_score_override": 0.30}},
    {"label": "borderline", "expected_set": {"CLARIFICATION_NEEDED", "REVIEW", "APPROVE"}, "input": {
        "user_id": "user_001", "amount": 350.00, "merchant_category": "Clothing",
        "time_of_day": "08:00 PM", "city": "Nashville, TN", "fraud_score_override": 0.25}},
    {"label": "borderline", "expected_set": {"REVIEW", "CLARIFICATION_NEEDED", "BLOCK"}, "input": {
        "user_id": "user_001", "amount": 600.00, "merchant_category": "Electronics",
        "time_of_day": "11:30 PM", "city": "Memphis, TN", "fraud_score_override": 0.42}},

    # ── IMPOSSIBLE TRAVEL (should BLOCK) ──
    {"label": "impossible_travel", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 500.00, "merchant_category": "Hotel",
        "time_of_day": "06:00 AM", "city": "Tokyo, Japan", "fraud_score_override": 0.75}},
    {"label": "impossible_travel", "expected": "BLOCK", "input": {
        "user_id": "user_001", "amount": 300.00, "merchant_category": "Restaurant",
        "time_of_day": "11:00 PM", "city": "London, UK", "fraud_score_override": 0.70}},

    # ── NORMAL SMALL SPENDER (should APPROVE or CLARIFY) ──
    {"label": "normal_small_spender", "expected_set": {"APPROVE", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_001", "amount": 156.20, "merchant_category": "Gas Station",
        "time_of_day": "05:15 PM", "city": "Nashville, TN", "fraud_score_override": 0.15}},
    {"label": "normal_small_spender", "expected_set": {"APPROVE", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_001", "amount": 89.99, "merchant_category": "Clothing",
        "time_of_day": "02:30 PM", "city": "Nashville, TN", "fraud_score_override": 0.08}},
    {"label": "normal_small_spender", "expected_set": {"APPROVE", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_001", "amount": 35.00, "merchant_category": "Grocery",
        "time_of_day": "06:00 PM", "city": "Nashville, TN", "fraud_score_override": 0.05}},

    # ── EDGE CASES ──
    {"label": "round_number", "expected_set": {"BLOCK", "REVIEW", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_001", "amount": 1000.00, "merchant_category": "Wire Transfer",
        "time_of_day": "01:00 AM", "city": "Nashville, TN", "fraud_score_override": 0.65}},
    {"label": "gift_card", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_001", "amount": 500.00, "merchant_category": "Gift Cards",
        "time_of_day": "12:15 AM", "city": "Nashville, TN", "fraud_score_override": 0.70}},
    {"label": "crypto", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_001", "amount": 2000.00, "merchant_category": "Cryptocurrency Exchange",
        "time_of_day": "03:30 AM", "city": "Nashville, TN", "fraud_score_override": 0.82}},
    {"label": "daytime_normal", "expected_set": {"APPROVE", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_001", "amount": 75.00, "merchant_category": "Restaurant",
        "time_of_day": "12:30 PM", "city": "Nashville, TN", "fraud_score_override": 0.12}},
    {"label": "weekend_shopping", "expected_set": {"APPROVE", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_001", "amount": 200.00, "merchant_category": "Retail",
        "time_of_day": "03:00 PM", "city": "Nashville, TN", "fraud_score_override": 0.18}},

    # ── MIXED SIGNALS ──
    {"label": "mixed_signals", "expected_set": {"REVIEW", "CLARIFICATION_NEEDED", "BLOCK"}, "input": {
        "user_id": "user_001", "amount": 400.00, "merchant_category": "Electronics",
        "time_of_day": "10:00 PM", "city": "Nashville, TN", "fraud_score_override": 0.35}},
    {"label": "mixed_signals", "expected_set": {"REVIEW", "CLARIFICATION_NEEDED", "APPROVE"}, "input": {
        "user_id": "user_002", "amount": 5000.00, "merchant_category": "Jewelry",
        "time_of_day": "08:00 PM", "city": "San Francisco, CA", "fraud_score_override": 0.50}},
    {"label": "mixed_signals", "expected_set": {"REVIEW", "CLARIFICATION_NEEDED", "BLOCK"}, "input": {
        "user_id": "user_001", "amount": 750.00, "merchant_category": "Gambling",
        "time_of_day": "11:00 PM", "city": "Las Vegas, NV", "fraud_score_override": 0.55}},

    # ── LARGE AMOUNT, NORMAL CONTEXT ──
    {"label": "large_normal", "expected_set": {"APPROVE", "REVIEW", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_002", "amount": 4500.00, "merchant_category": "Electronics",
        "time_of_day": "01:00 PM", "city": "San Francisco, CA", "fraud_score_override": 0.42}},
    {"label": "large_normal", "expected_set": {"APPROVE", "REVIEW", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_002", "amount": 3800.00, "merchant_category": "Apple Store",
        "time_of_day": "03:20 PM", "city": "San Francisco, CA", "fraud_score_override": 0.35}},

    # ── ZERO HISTORY USER ──
    {"label": "unknown_user", "expected_set": {"REVIEW", "BLOCK", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_999", "amount": 500.00, "merchant_category": "Electronics",
        "time_of_day": "02:00 AM", "city": "Dallas, TX", "fraud_score_override": 0.65}},
    {"label": "unknown_user", "expected_set": {"REVIEW", "CLARIFICATION_NEEDED"}, "input": {
        "user_id": "user_999", "amount": 100.00, "merchant_category": "Restaurant",
        "time_of_day": "07:00 PM", "city": "Austin, TX", "fraud_score_override": 0.20}},

    # ── DUPLICATE / REPLAY ──
    {"label": "duplicate_pattern", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_003", "amount": 1.00, "merchant_category": "Online Retail",
        "time_of_day": "07:11 AM", "city": "Chicago, IL", "fraud_score_override": 0.62}},
    {"label": "duplicate_pattern", "expected_set": {"BLOCK", "REVIEW"}, "input": {
        "user_id": "user_003", "amount": 0.99, "merchant_category": "Online Retail",
        "time_of_day": "07:12 AM", "city": "Chicago, IL", "fraud_score_override": 0.60}},
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation(base_url: str) -> dict:
    results = []
    latencies = []
    passed = 0
    total = len(TEST_CASES)

    print(f"\nRunning {total} test cases against {base_url}/analyze")
    print("=" * 65)

    for i, tc in enumerate(TEST_CASES, 1):
        try:
            t0 = time.time()
            resp = requests.post(f"{base_url}/analyze", json=tc["input"], timeout=60)
            latency_ms = int((time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [{i:02d}] ERROR {tc['label']}: {e}")
            results.append({**tc, "actual": "ERROR", "pass": False, "latency_ms": 0})
            continue

        actual = data.get("decision", "UNKNOWN")
        latencies.append(latency_ms)

        # Check pass/fail
        if "expected" in tc:
            ok = actual == tc["expected"]
        else:
            ok = actual in tc["expected_set"]

        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        routed = data.get("routed_to", "?")
        score = data.get("fraud_score", 0)

        expected_str = tc.get("expected") or "/".join(sorted(tc.get("expected_set", [])))
        print(f"  [{i:02d}] {status} | {tc['label']:<22} | got={actual:<22} expected={expected_str:<30} | score={score:.4f} route={routed} {latency_ms}ms")

        results.append({
            "label": tc["label"],
            "actual": actual,
            "expected": expected_str,
            "pass": ok,
            "fraud_score": score,
            "routed_to": routed,
            "latency_ms": latency_ms,
        })

    print("=" * 65)

    # Summary stats
    accuracy = passed / total
    decision_counts = {}
    routing_counts = {}
    for r in results:
        decision_counts[r["actual"]] = decision_counts.get(r["actual"], 0) + 1
        routing_counts[r.get("routed_to", "?")] = routing_counts.get(r.get("routed_to", "?"), 0) + 1

    avg_lat = statistics.mean(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    summary = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "base_url": base_url,
        "total": total,
        "passed": passed,
        "accuracy": round(accuracy, 4),
        "decision_distribution": decision_counts,
        "routing_distribution": routing_counts,
        "latency_avg_ms": round(avg_lat),
        "latency_p95_ms": round(p95_lat),
        "results": results,
    }

    print(f"\nAccuracy     : {passed}/{total}  ({accuracy:.1%})")
    print(f"Decisions    : {decision_counts}")
    print(f"Routing      : {routing_counts}")
    print(f"Latency avg  : {round(avg_lat)}ms   p95: {round(p95_lat)}ms")

    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results → {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()
    run_evaluation(args.url)
