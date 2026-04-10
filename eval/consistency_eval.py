"""
Consistency & explanation quality evaluation.

Runs the same 6 transactions N times each and checks:
  1. Decision consistency (does the same input always get the same decision?)
  2. Score stability (does the fraud score vary across runs?)
  3. Explanation grounding (does the explanation mention actual signal keywords?)
  4. Unknown-user hallucination (does the agent invent history for user_999?)

Usage:
    python consistency_eval.py --url https://wysewao87f.execute-api.us-east-2.amazonaws.com
    python consistency_eval.py --url https://wysewao87f.execute-api.us-east-2.amazonaws.com --runs 3
"""

import argparse
import json
import time
import requests
from collections import Counter

PROBE_CASES = [
    {
        "name": "high_fraud_night",
        "description": "High score (0.92), 1:45 AM, Jewelry — expect BLOCK",
        "input": {
            "user_id": "user_001", "amount": 1800.00,
            "merchant_category": "Jewelry", "time_of_day": "01:45 AM",
            "city": "New York, NY", "fraud_score_override": 0.92,
        },
        "expected": "BLOCK",
        "grounding_keywords": ["score", "night", "jewelry", "1:45", "unusual", "high"],
    },
    {
        "name": "legit_big_spender",
        "description": "High amount but known big spender (user_002) — expect APPROVE",
        "input": {
            "user_id": "user_002", "amount": 3500.00,
            "merchant_category": "Electronics", "time_of_day": "02:00 PM",
            "city": "San Francisco, CA", "fraud_score_override": 0.45,
        },
        "expected": "APPROVE",
        "grounding_keywords": ["consistent", "history", "normal", "pattern", "score"],
    },
    {
        "name": "borderline_review",
        "description": "Medium score (0.38), late night electronics — expect REVIEW or CLARIFICATION_NEEDED",
        "input": {
            "user_id": "user_001", "amount": 800.00,
            "merchant_category": "Electronics", "time_of_day": "11:00 PM",
            "city": "Nashville, TN", "fraud_score_override": 0.38,
        },
        "expected_set": {"REVIEW", "CLARIFICATION_NEEDED", "BLOCK"},
        "grounding_keywords": ["score", "night", "electronics", "unusual"],
    },
    {
        "name": "impossible_travel",
        "description": "Tokyo transaction for Nashville-based user — expect BLOCK",
        "input": {
            "user_id": "user_001", "amount": 500.00,
            "merchant_category": "Hotel", "time_of_day": "06:00 AM",
            "city": "Tokyo, Japan", "fraud_score_override": 0.75,
        },
        "expected": "BLOCK",
        "grounding_keywords": ["travel", "location", "tokyo", "geographic", "impossible", "distance"],
    },
    {
        "name": "unknown_user_high_risk",
        "description": "Unknown user (user_999), 2 AM, high score — should NOT invent history",
        "input": {
            "user_id": "user_999", "amount": 500.00,
            "merchant_category": "Electronics", "time_of_day": "02:00 AM",
            "city": "Dallas, TX", "fraud_score_override": 0.65,
        },
        "expected_set": {"REVIEW", "BLOCK", "CLARIFICATION_NEEDED"},
        "grounding_keywords": ["no history", "unknown", "no transaction", "no prior", "new user", "no record"],
        "hallucination_check": True,
    },
    {
        "name": "auto_approve_fast",
        "description": "Very low score (0.0001) — should bypass LLM entirely",
        "input": {
            "user_id": "user_001", "amount": 23.50,
            "merchant_category": "Grocery", "time_of_day": "08:30 AM",
            "city": "Nashville, TN", "fraud_score_override": 0.0001,
        },
        "expected_set": {"APPROVE", "APPROVED"},
        "grounding_keywords": [],
        "check_route": "auto_approve",
    },
]


def call_api(url: str, payload: dict, timeout: int = 60) -> tuple[dict, int]:
    t0 = time.time()
    resp = requests.post(f"{url}/analyze", json=payload, timeout=timeout)
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    return resp.json(), latency_ms


def check_grounding(explanation: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Return (grounded, matched_keywords)."""
    if not keywords:
        return True, []
    lower = explanation.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    return len(matched) > 0, matched


def run_consistency(base_url: str, runs: int):
    print(f"\nConsistency & Explanation Eval — {runs} runs per case")
    print(f"Endpoint: {base_url}")
    print("=" * 70)

    all_results = {}

    for case in PROBE_CASES:
        print(f"\n[{case['name']}] {case['description']}")
        decisions = []
        scores = []
        latencies = []
        grounded_count = 0
        hallucination_flags = []
        routes = []

        for run_i in range(runs):
            try:
                data, latency_ms = call_api(base_url, case["input"])
            except Exception as e:
                print(f"  Run {run_i+1}: ERROR — {e}")
                continue

            decision = data.get("decision", "UNKNOWN")
            score = data.get("fraud_score", 0)
            explanation = data.get("explanation", "")
            route = data.get("routed_to", "?")

            decisions.append(decision)
            scores.append(score)
            latencies.append(latency_ms)
            routes.append(route)

            grounded, matched = check_grounding(explanation, case.get("grounding_keywords", []))
            if grounded:
                grounded_count += 1

            # Hallucination check: for unknown users, flag if explanation sounds
            # like it's describing real transaction history
            hall_flag = False
            if case.get("hallucination_check"):
                invention_phrases = [
                    "previous transactions", "transaction history shows",
                    "based on their history", "past purchases", "regularly shops",
                    "typically spends", "usual pattern"
                ]
                lower_exp = explanation.lower()
                hall_flag = any(p in lower_exp for p in invention_phrases)
                hallucination_flags.append(hall_flag)

            status = "OK" if grounded or not case.get("grounding_keywords") else "WEAK"
            hall_str = " [HALLUCINATION?]" if hall_flag else ""
            print(f"  Run {run_i+1}: {decision:<22} score={score:.4f} {latency_ms}ms route={route} explanation={status}{hall_str}")

        if not decisions:
            continue

        # Consistency
        decision_counts = Counter(decisions)
        most_common, most_common_count = decision_counts.most_common(1)[0]
        consistency_pct = most_common_count / len(decisions) * 100

        # Score range
        score_min, score_max = min(scores), max(scores)
        score_stable = (score_max - score_min) < 0.01  # scores should be deterministic

        # Expected pass
        if "expected" in case:
            passed = all(d == case["expected"] for d in decisions)
            expected_str = case["expected"]
        else:
            passed = all(d in case["expected_set"] for d in decisions)
            expected_str = "/".join(sorted(case["expected_set"]))

        # Route check
        route_ok = True
        if case.get("check_route"):
            route_ok = all(r == case["check_route"] for r in routes)

        print(f"  ── Consistency : {consistency_pct:.0f}%  ({dict(decision_counts)})")
        print(f"  ── Score range : {score_min:.4f}–{score_max:.4f}  {'STABLE' if score_stable else 'VARIABLE'}")
        print(f"  ── Grounded    : {grounded_count}/{runs} runs had relevant keywords")
        if hallucination_flags:
            hall_count = sum(hallucination_flags)
            print(f"  ── Hallucination: {hall_count}/{runs} runs invented history  {'PASS' if hall_count == 0 else 'FAIL'}")
        if case.get("check_route"):
            print(f"  ── Route check : {'PASS' if route_ok else 'FAIL'} (expected {case['check_route']})")
        print(f"  ── Decision OK : {'PASS' if passed else 'FAIL'}  (expected {expected_str})")

        all_results[case["name"]] = {
            "runs": runs,
            "decisions": decisions,
            "decision_distribution": dict(decision_counts),
            "consistency_pct": round(consistency_pct, 1),
            "scores": scores,
            "score_stable": score_stable,
            "grounded_pct": round(grounded_count / runs * 100, 1),
            "all_correct": passed,
            "latency_avg_ms": round(sum(latencies) / len(latencies)),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in all_results.items():
        cons = r["consistency_pct"]
        correct = "PASS" if r["all_correct"] else "FAIL"
        stable = "stable" if r["score_stable"] else "variable score"
        print(f"  {name:<25} consistency={cons:>5.1f}%  {correct}  {stable}  grounded={r['grounded_pct']}%")

    out_path = "consistency_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=3, help="How many times to run each case")
    args = parser.parse_args()
    run_consistency(args.url, args.runs)
