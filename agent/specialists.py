"""
Specialist agents — each focused on one fraud signal domain.
Each returns a structured risk report that the supervisor synthesizes.
"""

import os
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Shared LLM factory
# ---------------------------------------------------------------------------

def _llm():
    return ChatBedrock(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
    )


# ---------------------------------------------------------------------------
# Velocity Specialist
# ---------------------------------------------------------------------------

VELOCITY_PROMPT = """\
You are a transaction velocity analyst for a fraud detection system.
Your ONLY job is to assess whether the frequency of recent transactions is suspicious.

You will receive a transaction and the user's recent transaction history.
Analyze:
- How many transactions in the last 10-30 minutes?
- Is there a burst pattern typical of card testing (small amounts, rapid succession)?
- What is the normal transaction frequency for this user?

Respond in this exact format:
RISK_LEVEL: [LOW|MEDIUM|HIGH]
FINDING: [one clear sentence explaining your finding]
SIGNAL: [specific data point e.g. "4 transactions in 10 minutes"]
"""


def run_velocity_specialist(transaction: dict, velocity_data: str) -> dict:
    llm = _llm()
    messages = [
        SystemMessage(content=VELOCITY_PROMPT),
        HumanMessage(content=(
            f"Transaction: ${transaction['amount']:.2f} at {transaction['merchant_category']}, "
            f"{transaction['time_of_day']} for user {transaction['user_id']}\n\n"
            f"Velocity data:\n{velocity_data}"
        ))
    ]
    response = llm.invoke(messages)
    return _parse_report("velocity", response.content)


# ---------------------------------------------------------------------------
# Location Specialist
# ---------------------------------------------------------------------------

LOCATION_PROMPT = """\
You are a geographic fraud analyst for a fraud detection system.
Your ONLY job is to assess whether the transaction location is suspicious.

You will receive the current transaction city and the user's last known location with time elapsed.
Analyze:
- Is travel between the two cities physically possible in the given time?
- Is this a known high-risk location pattern?
- Does the city match the user's established home region?

Consider: flights between US cities take 2-6 hours. Driving takes longer.
A transaction in New York followed 1 hour later by one in Los Angeles is impossible.

Respond in this exact format:
RISK_LEVEL: [LOW|MEDIUM|HIGH]
FINDING: [one clear sentence explaining your finding]
SIGNAL: [specific data point e.g. "Nashville → Los Angeles in 48 hours — feasible"]
"""


def run_location_specialist(transaction: dict, location_data: str) -> dict:
    llm = _llm()
    messages = [
        SystemMessage(content=LOCATION_PROMPT),
        HumanMessage(content=(
            f"Current city: {transaction.get('city', 'Unknown')}\n"
            f"User: {transaction['user_id']}\n\n"
            f"Location history data:\n{location_data}"
        ))
    ]
    response = llm.invoke(messages)
    return _parse_report("location", response.content)


# ---------------------------------------------------------------------------
# Spending Specialist
# ---------------------------------------------------------------------------

SPENDING_PROMPT = """\
You are a spending pattern analyst for a fraud detection system.
Your ONLY job is to assess whether the transaction amount is suspicious.

You will receive the current amount and the user's historical spending data.
Analyze:
- How does this amount compare to the user's average and maximum?
- Is the merchant category consistent with past behavior?
- Is the amount a round number (common in fraud)?

Respond in this exact format:
RISK_LEVEL: [LOW|MEDIUM|HIGH]
FINDING: [one clear sentence explaining your finding]
SIGNAL: [specific data point e.g. "19x above average spend of $65"]
"""


def run_spending_specialist(transaction: dict, spending_data: str) -> dict:
    llm = _llm()
    messages = [
        SystemMessage(content=SPENDING_PROMPT),
        HumanMessage(content=(
            f"Current transaction: ${transaction['amount']:.2f} at "
            f"{transaction['merchant_category']} for user {transaction['user_id']}\n\n"
            f"Spending history:\n{spending_data}"
        ))
    ]
    response = llm.invoke(messages)
    return _parse_report("spending", response.content)


# ---------------------------------------------------------------------------
# Temporal Specialist
# ---------------------------------------------------------------------------

TEMPORAL_PROMPT = """\
You are a temporal pattern analyst for a fraud detection system.
Your ONLY job is to assess whether the time of this transaction is suspicious.

You will receive the current transaction time and the user's typical transaction hours.
Analyze:
- Is this time outside the user's normal transaction window?
- Is this a time commonly associated with fraud (2AM-5AM)?
- Is the time consistent with the claimed merchant category (e.g. grocery at 3AM is unusual)?

Respond in this exact format:
RISK_LEVEL: [LOW|MEDIUM|HIGH]
FINDING: [one clear sentence explaining your finding]
SIGNAL: [specific data point e.g. "2:14 AM — outside normal window of 08:00-19:00"]
"""


def run_temporal_specialist(transaction: dict, temporal_data: str) -> dict:
    llm = _llm()
    messages = [
        SystemMessage(content=TEMPORAL_PROMPT),
        HumanMessage(content=(
            f"Current time: {transaction['time_of_day']}\n"
            f"Merchant: {transaction['merchant_category']}\n"
            f"User: {transaction['user_id']}\n\n"
            f"Temporal pattern data:\n{temporal_data}"
        ))
    ]
    response = llm.invoke(messages)
    return _parse_report("temporal", response.content)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_report(domain: str, text: str) -> dict:
    """Parse the structured specialist response into a dict."""
    lines = {
        line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
        for line in text.strip().splitlines()
        if ":" in line
    }
    return {
        "domain": domain,
        "risk_level": lines.get("RISK_LEVEL", "MEDIUM"),
        "finding": lines.get("FINDING", text[:200]),
        "signal": lines.get("SIGNAL", ""),
        "raw": text,
    }
