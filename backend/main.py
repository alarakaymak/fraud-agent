"""
FastAPI backend for the Fraud Detection Agent.
Local dev: uvicorn main:app --reload --port 8000
Lambda:    see lambda_function.py
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fraud_agent")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Make agent/ importable from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

from agent import agent
from scorer import score, route, scale_features

app = FastAPI(title="Fraud Detection Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    user_id: str = Field(..., example="user_001")
    amount: float = Field(..., gt=0, example=1250.00)
    merchant_category: str = Field(..., example="Electronics")
    time_of_day: str = Field(..., example="02:14 AM")
    city: str = Field(default="Unknown", example="Los Angeles, CA")
    fraud_score_override: float | None = Field(default=None, ge=0.0, le=1.0, example=0.73)
    features: dict | None = Field(default=None)


class TransactionResponse(BaseModel):
    transaction_id: str
    decision: str
    explanation: str
    fraud_score: float
    routed_to: str          # "auto_approve" | "agent"
    latency_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_synthetic_features(amount: float, time_of_day: str) -> dict:
    """
    Build a feature dict for the classifier using Amount and Time.
    V1-V28 default to 0 (neutral PCA components).
    """
    import re

    match = re.match(r"(\d+):(\d+)\s*(AM|PM)", time_of_day.strip().upper())
    if match:
        h, m, period = int(match.group(1)), int(match.group(2)), match.group(3)
        if period == "PM" and h != 12:
            h += 12
        if period == "AM" and h == 12:
            h = 0
        time_seconds = h * 3600 + m * 60
    else:
        time_seconds = 43200  # default noon

    amount_scaled, time_scaled = scale_features(amount, time_seconds)

    features = {f"V{i}": 0.0 for i in range(1, 29)}
    features["Amount_scaled"] = amount_scaled
    features["Time_scaled"] = time_scaled
    return features


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the frontend UI — same origin, no CORS needed."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.post("/analyze", response_model=TransactionResponse)
def analyze_transaction(req: TransactionRequest):
    start = time.time()

    transaction_id = f"txn_{uuid.uuid4().hex[:8]}"

    # Score: use override if provided, else run classifier on features
    if req.fraud_score_override is not None:
        fraud_score = req.fraud_score_override
    else:
        features = req.features or build_synthetic_features(req.amount, req.time_of_day)
        try:
            fraud_score = score(features)
        except Exception as e:
            log.error(json.dumps({"event": "classifier_error", "transaction_id": transaction_id, "error": str(e)}))
            raise HTTPException(status_code=500, detail=f"Classifier error: {e}")

    routing = route(fraud_score)

    transaction = {
        "transaction_id": transaction_id,
        "user_id": req.user_id,
        "amount": req.amount,
        "merchant_category": req.merchant_category,
        "time_of_day": req.time_of_day,
        "city": req.city,
        "fraud_score": fraud_score,
    }

    log.info(json.dumps({
        "event": "request",
        "transaction_id": transaction_id,
        "user_id": req.user_id,
        "amount": req.amount,
        "merchant_category": req.merchant_category,
        "city": req.city,
        "fraud_score": round(fraud_score, 6),
        "routing": routing,
    }))

    if routing == "auto_approve":
        decision = "APPROVED"
        explanation = f"Auto-approved: fraud score {fraud_score:.5f} is below threshold."
    else:
        try:
            result = agent.invoke({
                "transaction": transaction,
                "messages": [],
                "velocity_report": {},
                "location_report": {},
                "spending_report": {},
                "temporal_report": {},
                "decision": "",
                "explanation": "",
            })
            decision = result["decision"]
            explanation = result["explanation"]
        except Exception as e:
            log.error(json.dumps({"event": "agent_error", "transaction_id": transaction_id, "error": str(e)}))
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    latency_ms = int((time.time() - start) * 1000)

    log.info(json.dumps({
        "event": "response",
        "transaction_id": transaction_id,
        "user_id": req.user_id,
        "decision": decision,
        "fraud_score": round(fraud_score, 6),
        "routed_to": routing,
        "latency_ms": latency_ms,
    }))

    return TransactionResponse(
        transaction_id=transaction_id,
        decision=decision,
        explanation=explanation,
        fraud_score=round(fraud_score, 6),
        routed_to=routing,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# /reply — multi-turn cardholder conversation
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: str   # "agent" | "user"
    content: str


class ReplyRequest(BaseModel):
    transaction_id: str
    user_reply: str
    transaction: TransactionRequest
    history: list[ConversationTurn] = []   # full conversation so far


class ReplyResponse(BaseModel):
    agent_message: str          # what the agent says next
    decision: str | None        # None = still conversing, set when resolved
    explanation: str | None
    latency_ms: int


REPLY_SYSTEM_PROMPT = """\
You are a fraud detection agent for a bank. A suspicious transaction has been flagged \
and you are having a live conversation with the cardholder to verify it.

Your job:
1. Understand the cardholder's response naturally — they may be confused, uncertain, or give partial answers.
2. Ask targeted follow-up questions if needed (e.g. location, what they bought, who made the purchase).
3. When you have enough confidence, make a final decision:
   - If they clearly confirmed it: respond warmly, say the transaction is approved.
   - If they clearly denied it: say you're blocking the transaction and their card for safety.
   - If still uncertain after 2+ exchanges: say you're escalating to a human specialist.
4. Keep responses short (2-3 sentences max). Sound like a helpful bank assistant, not a robot.
5. IMPORTANT: Always end your response with one of these exact tags on a new line:
   [DECISION: APPROVE] — cardholder confirmed
   [DECISION: BLOCK] — cardholder denied
   [DECISION: REVIEW] — too uncertain, escalating to human
   [CONTINUE] — still gathering information, ask a follow-up\
"""


@app.post("/reply", response_model=ReplyResponse)
def reply_to_clarification(req: ReplyRequest):
    start = time.time()
    fraud_score = req.transaction.fraud_score_override or 0.5

    try:
        from langchain_aws import ChatBedrock
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        from tools import log_decision

        # Build conversation history for the LLM
        messages = [SystemMessage(content=REPLY_SYSTEM_PROMPT)]

        # Add transaction context as first human message
        context = (
            f"Transaction flagged: ${req.transaction.amount:.2f} at "
            f"{req.transaction.merchant_category}, {req.transaction.time_of_day}. "
            f"Classifier fraud score: {fraud_score:.4f}. User: {req.transaction.user_id}."
        )
        messages.append(HumanMessage(content=f"[CONTEXT] {context}"))
        messages.append(AIMessage(content="Understood. I'll handle this verification."))

        # Replay conversation history
        for turn in req.history:
            if turn.role == "agent":
                messages.append(AIMessage(content=turn.content))
            else:
                messages.append(HumanMessage(content=turn.content))

        # Add the latest user reply
        messages.append(HumanMessage(content=req.user_reply))

        # Call LLM directly — no graph, avoids SystemMessage conflicts
        llm = ChatBedrock(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        )
        response = llm.invoke(messages)
        agent_text = response.content

        # Parse decision tag from response
        decision = None
        explanation = None
        agent_message = agent_text

        import re
        decision_match = re.search(r"\[DECISION:\s*(APPROVE|BLOCK|REVIEW)\]", agent_text)
        continue_match = re.search(r"\[CONTINUE\]", agent_text)

        if decision_match:
            decision = decision_match.group(1)
            explanation = re.sub(r"\[DECISION:[^\]]+\]", "", agent_text).strip()
            agent_message = explanation
            log_decision(req.transaction_id, req.transaction.user_id, fraud_score,
                        decision, explanation, ["request_clarification", "multi_turn_reply"])
        elif continue_match:
            agent_message = re.sub(r"\[CONTINUE\]", "", agent_text).strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    latency_ms = int((time.time() - start) * 1000)
    return ReplyResponse(
        agent_message=agent_message,
        decision=decision,
        explanation=explanation,
        latency_ms=latency_ms,
    )
