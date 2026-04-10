"""
Fraud Detection — Supervisor Multi-Agent System
------------------------------------------------
Graph flow:
  START → route_node
    ├── auto_approve (score < threshold) → END
    └── dispatch → runs 4 specialist agents in parallel
                        ↓
                   synthesize_node (supervisor decides)
                        ↓
                   END  or  CLARIFICATION_NEEDED → END
"""

import json
import operator
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Sequence, TypedDict

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tools import (
    ALL_TOOLS,
    check_velocity,
    check_location,
    analyze_spending_pattern,
    check_temporal_pattern,
    lookup_user_history,
    log_decision,
)
from scorer import AUTO_APPROVE_THRESHOLD
from specialists import (
    run_velocity_specialist,
    run_location_specialist,
    run_spending_specialist,
    run_temporal_specialist,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FraudState(TypedDict):
    transaction: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Specialist findings (populated in parallel)
    velocity_report: dict
    location_report: dict
    spending_report: dict
    temporal_report: dict

    # Final output
    decision: str
    explanation: str


# ---------------------------------------------------------------------------
# Supervisor LLM (synthesizes specialist findings)
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """\
You are the lead fraud detection supervisor at a financial institution.

Four specialist analysts have independently investigated a suspicious transaction \
and each produced a risk report. Your job is to:

1. Read all four specialist reports carefully.
2. Weigh the signals together — multiple HIGH signals = strong fraud evidence.
3. Make ONE final decision from these options:
   - BLOCK       : Strong converging evidence of fraud (2+ HIGH signals, or 1 HIGH + classifier score > 0.8)
   - APPROVE     : Evidence clearly supports legitimacy (all LOW/MEDIUM, plausible explanations)
   - REVIEW      : Mixed signals, cannot decide confidently — escalate to human
   - CLARIFICATION_NEEDED : One targeted question to the cardholder would resolve ambiguity

4. Call the appropriate tool: block_transaction, approve_transaction, review_transaction, or request_clarification.

Be decisive. Explain your reasoning by referencing the specialist findings.\
"""

llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
).bind_tools([t for t in ALL_TOOLS if t.name in (
    "block_transaction", "approve_transaction",
    "review_transaction", "request_clarification"
)])

# Alias for external imports (backend reply endpoint)
SYSTEM_PROMPT = SUPERVISOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def route_node(state: FraudState) -> dict:
    """Auto-approve low-score transactions; dispatch the rest to specialists."""
    txn = state["transaction"]
    score = txn["fraud_score"]

    if score < AUTO_APPROVE_THRESHOLD:
        explanation = f"Auto-approved: fraud score {score:.5f} is below threshold {AUTO_APPROVE_THRESHOLD}."
        log_decision(txn["transaction_id"], txn["user_id"], score,
                     "APPROVE", explanation, [])
        return {"decision": "APPROVE", "explanation": explanation}

    return {}


def dispatch_node(state: FraudState) -> dict:
    """
    Fetch raw data for all specialists, then run them in parallel threads.
    Each specialist is a focused LLM call with 2-3 relevant data points.
    """
    txn = state["transaction"]
    user_id = txn["user_id"]

    # Fetch raw tool data (fast DynamoDB/mock calls)
    velocity_raw  = check_velocity.invoke({"user_id": user_id, "window_minutes": 10})
    location_raw  = check_location.invoke({"user_id": user_id, "current_city": txn.get("city", "Unknown")})
    spending_raw  = analyze_spending_pattern.invoke({"user_id": user_id, "current_amount": txn["amount"]})
    temporal_raw  = check_temporal_pattern.invoke({"user_id": user_id, "current_time_of_day": txn["time_of_day"]})

    # Run 4 specialist LLM calls in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_velocity_specialist, txn, velocity_raw):  "velocity",
            executor.submit(run_location_specialist, txn, location_raw):  "location",
            executor.submit(run_spending_specialist, txn, spending_raw):  "spending",
            executor.submit(run_temporal_specialist, txn, temporal_raw):  "temporal",
        }
        for future in as_completed(futures):
            domain = futures[future]
            try:
                results[domain] = future.result()
            except Exception as e:
                results[domain] = {
                    "domain": domain, "risk_level": "MEDIUM",
                    "finding": f"Analysis failed: {e}", "signal": "", "raw": str(e)
                }

    return {
        "velocity_report": results["velocity"],
        "location_report": results["location"],
        "spending_report": results["spending"],
        "temporal_report": results["temporal"],
    }


def synthesize_node(state: FraudState) -> dict:
    """
    Supervisor reads all specialist reports and makes the final decision
    by calling one of the 4 decision tools.
    """
    txn = state["transaction"]
    reports = [
        state["velocity_report"],
        state["location_report"],
        state["spending_report"],
        state["temporal_report"],
    ]

    # Format specialist findings for the supervisor
    findings_text = "\n\n".join([
        f"=== {r['domain'].upper()} ANALYST ===\n"
        f"Risk Level : {r['risk_level']}\n"
        f"Finding    : {r['finding']}\n"
        f"Signal     : {r['signal']}"
        for r in reports
    ])

    high_count = sum(1 for r in reports if r["risk_level"] == "HIGH")
    medium_count = sum(1 for r in reports if r["risk_level"] == "MEDIUM")

    supervisor_prompt = (
        f"Transaction under review:\n"
        f"  User     : {txn['user_id']}\n"
        f"  Amount   : ${txn['amount']:.2f}\n"
        f"  Merchant : {txn['merchant_category']}\n"
        f"  Time     : {txn['time_of_day']}\n"
        f"  City     : {txn.get('city', 'Unknown')}\n"
        f"  ML Score : {txn['fraud_score']:.4f}\n\n"
        f"Specialist reports ({high_count} HIGH, {medium_count} MEDIUM risk signals):\n\n"
        f"{findings_text}\n\n"
        "Based on these findings, make your final decision now."
    )

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=supervisor_prompt),
    ])

    # Extract decision from tool call
    if hasattr(response, "tool_calls") and response.tool_calls:
        tc = response.tool_calls[0]
        tool_name = tc["name"]
        args = tc["args"]

        decision_map = {
            "block_transaction":   "BLOCK",
            "approve_transaction": "APPROVE",
            "review_transaction":  "REVIEW",
            "request_clarification": "CLARIFICATION_NEEDED",
        }
        decision = decision_map.get(tool_name, "REVIEW")
        explanation = args.get("reason") or args.get("question") or ""

        # Attach specialist summary to explanation
        specialist_summary = " | ".join([
            f"{r['domain'].upper()}: {r['risk_level']}" for r in reports
        ])
        explanation = f"{explanation}\n\n[Specialist signals: {specialist_summary}]"

    else:
        decision = "REVIEW"
        explanation = response.content or "Supervisor could not reach a decision."

    tools_called = ["check_velocity", "check_location",
                    "analyze_spending_pattern", "check_temporal_pattern"]
    log_decision(txn["transaction_id"], txn["user_id"], txn["fraud_score"],
                 decision, explanation, tools_called)

    return {"decision": decision, "explanation": explanation}


# ---------------------------------------------------------------------------
# Routing conditions
# ---------------------------------------------------------------------------

def after_route(state: FraudState) -> str:
    return END if state.get("decision") else "dispatch"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(FraudState)

    g.add_node("route",      route_node)
    g.add_node("dispatch",   dispatch_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"dispatch": "dispatch", END: END})
    g.add_edge("dispatch",   "synthesize")
    g.add_edge("synthesize", END)

    # LangSmith tracing is enabled automatically when these env vars are set:
    #   LANGCHAIN_TRACING_V2=true
    #   LANGCHAIN_API_KEY=<key>
    #   LANGCHAIN_PROJECT=fraud-agent
    return g.compile()


agent = build_graph()


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        {
            "label": "Auto-approve (low score)",
            "transaction": {
                "transaction_id": "txn_001",
                "user_id": "user_001",
                "amount": 23.50,
                "merchant_category": "Grocery",
                "time_of_day": "08:30 AM",
                "city": "Nashville, TN",
                "fraud_score": 0.00005,
            },
        },
        {
            "label": "High risk — suspicious (all signals should converge)",
            "transaction": {
                "transaction_id": "txn_002",
                "user_id": "user_001",
                "amount": 1250.00,
                "merchant_category": "Electronics",
                "time_of_day": "02:14 AM",
                "city": "Los Angeles, CA",
                "fraud_score": 0.73,
            },
        },
        {
            "label": "Big spender — normal pattern",
            "transaction": {
                "transaction_id": "txn_003",
                "user_id": "user_002",
                "amount": 3500.00,
                "merchant_category": "Electronics",
                "time_of_day": "02:00 PM",
                "city": "San Francisco, CA",
                "fraud_score": 0.45,
            },
        },
        {
            "label": "High velocity — card testing pattern (user_003)",
            "transaction": {
                "transaction_id": "txn_004",
                "user_id": "user_003",
                "amount": 12.00,
                "merchant_category": "Online Retail",
                "time_of_day": "07:09 AM",
                "city": "Chicago, IL",
                "fraud_score": 0.55,
            },
        },
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {case['label']}")
        print(f"{'='*60}")

        result = agent.invoke({
            "transaction": case["transaction"],
            "messages": [],
            "velocity_report": {},
            "location_report": {},
            "spending_report": {},
            "temporal_report": {},
            "decision": "",
            "explanation": "",
        })

        print(f"Decision    : {result['decision']}")
        print(f"Explanation : {result['explanation'][:300]}")
        if result.get("velocity_report"):
            print(f"\nSpecialist reports:")
            for domain in ["velocity", "location", "spending", "temporal"]:
                r = result.get(f"{domain}_report", {})
                print(f"  {domain.upper():10} [{r.get('risk_level','?'):6}] {r.get('signal','')}")
