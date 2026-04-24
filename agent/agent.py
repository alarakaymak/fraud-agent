"""
Fraud Detection — Supervisor Multi-Agent System
------------------------------------------------
Graph flow:
  START → route_node
    ├── auto_approve (score < threshold) → END
    └── triage_node  (LLM selects relevant specialists)
              ↓
         dispatch_node (runs selected specialists in parallel)
              ↓
         synthesize_node (supervisor decides)
              ↓
         END
"""

import json
import operator
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Sequence, TypedDict

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from tools import (
    ALL_TOOLS,
    check_velocity,
    check_location,
    analyze_spending_pattern,
    check_temporal_pattern,
    log_decision,
)
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

    # Triage decision — which specialists to run
    specialist_selection: dict   # e.g. {"velocity": True, "location": False, ...}
    triage_reasoning: str        # short explanation of why those specialists were chosen

    # Specialist findings (populated selectively)
    velocity_report: dict
    location_report: dict
    spending_report: dict
    temporal_report: dict

    # Final output
    decision: str
    explanation: str


# ---------------------------------------------------------------------------
# Triage LLM — decides which specialists to dispatch
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are a fraud detection triage specialist. Given a transaction, decide which \
specialist analysts should investigate it.

Available specialists:
- velocity  : Detects burst patterns — many transactions in a short window (card testing).
              Relevant when: online retail, low amounts, score > 0.4, repeated merchants.
- location  : Detects impossible travel or new cities inconsistent with user history.
              Relevant when: unfamiliar city, travel merchants, recent location change.
- spending  : Detects anomalous amounts relative to the user's historical average.
              Relevant when: large purchases, unusual merchant category, any significant amount.
- temporal  : Detects unusual transaction times for this specific user.
              Relevant when: late-night transactions (10 PM–6 AM), weekend vs weekday patterns.

Rules:
- Always include spending — amount context is almost always relevant.
- Include at least 2 specialists total.
- Do not include specialists that add no signal for this transaction type.

Respond with ONLY a JSON object and a one-sentence reason, like:
{"selection": {"velocity": true, "location": false, "spending": true, "temporal": true}, \
"reason": "Late-night timing and high amount are the main signals; location is same city."}\
"""

triage_llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
)


# ---------------------------------------------------------------------------
# Supervisor LLM — synthesizes specialist findings into a final decision
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """\
You are the lead fraud detection supervisor at a financial institution.

Specialist analysts have investigated a suspicious transaction and produced risk reports. \
Not all specialists may have been dispatched — only those relevant to this transaction type \
were selected by the triage system. Focus on the reports provided.

Your job:
1. Read all specialist reports carefully.
2. Weigh the signals together — multiple HIGH signals = strong fraud evidence.
3. Make ONE final decision:
   - BLOCK                : Strong converging evidence of fraud. MUST use when 2 or more specialists \
report HIGH risk, regardless of score. Also use when score > 0.8 with any HIGH signal.
   - APPROVE              : All active specialist signals are LOW, score < 0.6, and the transaction \
fits the user's established pattern. Skipped specialists were irrelevant — their absence is not \
a reason to withhold approval.
   - CLARIFICATION_NEEDED : Ambiguous — the cardholder can resolve it with one question. Use when \
the transaction is in a known or home city, score is moderate (0.30–0.65), and the HIGH signal \
is spending or temporal (unusual amount or unusual time the cardholder can confirm). \
Do NOT use if the HIGH signal is velocity — a burst of transactions cannot be explained by \
the cardholder and should go to REVIEW instead.
   - REVIEW               : Conflicting signals with no clear resolution, HIGH velocity signal, \
unknown user, or multiple MEDIUM/HIGH signals without a single obvious question to ask.

   IMPORTANT: LOW signals from specialists reflect what the data shows — but if a specialist \
reports LOW simply because the user has no transaction history, treat that as uncertain, \
not as confirmed legitimacy. When history is absent, prefer REVIEW over APPROVE.

4. Call the appropriate tool: block_transaction, approve_transaction, review_transaction, \
or request_clarification.

Be decisive. Reference the specific specialist findings in your explanation.\
"""

supervisor_llm = ChatBedrock(
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
    """Routing is handled by main.py before reaching the agent; always triage."""
    return {}


def triage_node(state: FraudState) -> dict:
    """
    LLM-driven triage: decide which specialist agents are relevant for this
    transaction type. Uses Claude Haiku for speed.
    """
    txn = state["transaction"]
    prompt = (
        f"Transaction: ${txn['amount']:.2f} at {txn['merchant_category']}, "
        f"{txn['time_of_day']}, city: {txn.get('city', 'Unknown')}. "
        f"User: {txn['user_id']}. ML fraud score: {txn['fraud_score']:.4f}."
    )

    try:
        response = triage_llm.invoke([
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        selection = parsed.get("selection", {})
        reason = parsed.get("reason", "")
    except Exception:
        selection = {}
        reason = "Triage failed — defaulting to all specialists."

    # Spending always runs; enforce minimum of 2 specialists
    selection["spending"] = True
    if sum(1 for v in selection.values() if v) < 2:
        selection = {"velocity": True, "location": True, "spending": True, "temporal": True}
        reason = reason or "Minimum specialist threshold enforced."

    return {"specialist_selection": selection, "triage_reasoning": reason}


def dispatch_node(state: FraudState) -> dict:
    """
    Run only the specialists selected by triage, in parallel.
    """
    txn = state["transaction"]
    user_id = txn["user_id"]
    selection = state.get("specialist_selection", {
        "velocity": True, "location": True, "spending": True, "temporal": True
    })

    # Fetch raw DynamoDB/tool data only for selected specialists
    raw_data = {}
    if selection.get("velocity"):
        raw_data["velocity"] = check_velocity.invoke({"user_id": user_id, "window_minutes": 60})
    if selection.get("location"):
        raw_data["location"] = check_location.invoke({"user_id": user_id, "current_city": txn.get("city", "Unknown")})
    if selection.get("spending"):
        raw_data["spending"] = analyze_spending_pattern.invoke({"user_id": user_id, "current_amount": txn["amount"]})
    if selection.get("temporal"):
        raw_data["temporal"] = check_temporal_pattern.invoke({"user_id": user_id, "current_time_of_day": txn["time_of_day"]})

    specialist_runners = {
        "velocity": run_velocity_specialist,
        "location": run_location_specialist,
        "spending": run_spending_specialist,
        "temporal": run_temporal_specialist,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(specialist_runners[domain], txn, raw_data[domain]): domain
            for domain in raw_data
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
        "velocity_report": results.get("velocity", {}),
        "location_report": results.get("location", {}),
        "spending_report": results.get("spending", {}),
        "temporal_report": results.get("temporal", {}),
    }


def synthesize_node(state: FraudState) -> dict:
    """
    Supervisor reads specialist reports and makes the final decision.
    Only considers reports that were actually produced (non-empty).
    """
    txn = state["transaction"]

    # Filter to only the reports that were actually run
    all_reports = {
        "velocity": state.get("velocity_report", {}),
        "location": state.get("location_report", {}),
        "spending": state.get("spending_report", {}),
        "temporal": state.get("temporal_report", {}),
    }
    active_reports = {k: v for k, v in all_reports.items() if v}
    reports = list(active_reports.values())

    skipped = [k for k, v in all_reports.items() if not v]

    findings_text = "\n\n".join([
        f"=== {r['domain'].upper()} ANALYST ===\n"
        f"Risk Level : {r['risk_level']}\n"
        f"Finding    : {r['finding']}\n"
        f"Signal     : {r['signal']}"
        for r in reports
    ])

    if skipped:
        findings_text += f"\n\n[Triage deemed these dimensions irrelevant for this transaction type and skipped them: {', '.join(skipped)}. Treat skipped specialists as not applicable, not as unknown risk.]"

    high_count   = sum(1 for r in reports if r.get("risk_level") == "HIGH")
    medium_count = sum(1 for r in reports if r.get("risk_level") == "MEDIUM")

    supervisor_prompt = (
        f"Transaction under review:\n"
        f"  User     : {txn['user_id']}\n"
        f"  Amount   : ${txn['amount']:.2f}\n"
        f"  Merchant : {txn['merchant_category']}\n"
        f"  Time     : {txn['time_of_day']}\n"
        f"  City     : {txn.get('city', 'Unknown')}\n"
        f"  ML Score : {txn['fraud_score']:.4f}\n\n"
        f"Triage selected {len(reports)} specialist(s) "
        f"({high_count} HIGH, {medium_count} MEDIUM risk signals):\n\n"
        f"{findings_text}\n\n"
        "Based on these findings, make your final decision now."
    )

    response = supervisor_llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=supervisor_prompt),
    ])

    if hasattr(response, "tool_calls") and response.tool_calls:
        tc = response.tool_calls[0]
        decision_map = {
            "block_transaction":     "BLOCK",
            "approve_transaction":   "APPROVE",
            "review_transaction":    "REVIEW",
            "request_clarification": "CLARIFICATION_NEEDED",
        }
        decision    = decision_map.get(tc["name"], "REVIEW")
        explanation = tc["args"].get("reason") or tc["args"].get("question") or ""

        specialist_summary = " | ".join([
            f"{r['domain'].upper()}: {r['risk_level']}" for r in reports
        ])
        explanation = f"{explanation}\n\n[Specialist signals: {specialist_summary}]"
    else:
        decision    = "REVIEW"
        explanation = response.content or "Supervisor could not reach a decision."

    tools_called = [f"check_{d}" if d != "spending" else "analyze_spending_pattern"
                    for d in active_reports]
    log_decision(txn["transaction_id"], txn["user_id"], txn["fraud_score"],
                 decision, explanation, tools_called)

    return {"decision": decision, "explanation": explanation}


# ---------------------------------------------------------------------------
# Routing conditions
# ---------------------------------------------------------------------------

def after_route(state: FraudState) -> str:
    return END if state.get("decision") else "triage"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(FraudState)

    g.add_node("route",      route_node)
    g.add_node("triage",     triage_node)
    g.add_node("dispatch",   dispatch_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"triage": "triage", END: END})
    g.add_edge("triage",    "dispatch")
    g.add_edge("dispatch",  "synthesize")
    g.add_edge("synthesize", END)

    # LangSmith tracing is enabled automatically when these env vars are set:
    #   LANGCHAIN_TRACING_V2=true
    #   LANGCHAIN_API_KEY=<key>
    #   LANGCHAIN_PROJECT=fraud-agent
    return g.compile()


agent = build_graph()
