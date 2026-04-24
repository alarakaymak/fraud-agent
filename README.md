# FraudGuard — AI Fraud Detection Agent

Final project for DS5730 (Generative AI in Practice) at Vanderbilt University.

**Live demo**: https://wysewao87f.execute-api.us-east-2.amazonaws.com

## Overview

FraudGuard is a multi-agent fraud detection system that combines a supervised ML classifier with a LangGraph supervisor agent to analyze financial transactions in real time. The system produces a structured decision (APPROVE / REVIEW / BLOCK / CLARIFICATION_NEEDED) backed by four parallel specialist agents and an XGBoost fraud-probability score.

## Architecture

```
Transaction Input
       │
       ▼
 XGBoost Classifier  ──► Calibrated Fraud Score (0–1)
       │
       ▼
  LangGraph Supervisor
  ┌────┴───────────────────────────────┐
  │  Velocity     Location    Spending │
  │  Agent        Agent       Agent   │
  │                 Temporal          │
  │                 Agent             │
  └────────────────────────────────────┘
       │
       ▼
  Final Decision + Explanation
```

- **Supervisor**: Claude 3.5 Sonnet via Amazon Bedrock. Orchestrates specialist agents and synthesizes the final decision.
- **Velocity Agent**: Detects rapid successive transactions from the same user (burst patterns).
- **Location Agent**: Flags impossible travel — transactions physically separated beyond what travel time allows.
- **Spending Agent**: Identifies anomalous spend amounts relative to a user's historical behavior.
- **Temporal Agent**: Catches unusual transaction timing (e.g., 3 AM activity, holiday spikes).
- **ML Classifier**: XGBoost trained on transaction features, calibrated with IsotonicRegression. Provides a probability score independent of the LLM agents.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (supervisor pattern) |
| LLM | Claude 3.5 Sonnet (Amazon Bedrock) |
| ML model | XGBoost + IsotonicRegression |
| API | FastAPI + Mangum (Lambda adapter) |
| Compute | AWS Lambda (container image) |
| Storage | Amazon DynamoDB (transaction history) |
| Hosting | API Gateway HTTP API |
| Frontend | Vanilla HTML/CSS/JS (served from Lambda) |

## Project Structure

```
fraud-agent/
├── agent/
│   ├── agent.py          # LangGraph supervisor + specialist agents
│   ├── specialists.py    # Velocity, Location, Spending, Temporal agents
│   ├── scorer.py         # XGBoost model loader and scoring
│   ├── tools.py          # DynamoDB query tools
│   ├── setup_dynamodb.py # Table bootstrap script
│   └── requirements.txt
├── backend/
│   ├── main.py           # FastAPI app (REST API + UI serving)
│   ├── lambda_function.py # Mangum handler entry point
│   └── requirements.txt
├── classifier/           # Trained model artifacts (fraud_model.pkl)
├── frontend/
│   └── index.html        # Two-column dashboard UI
├── eval/
│   └── evaluate.py       # 40-case evaluation suite
├── Dockerfile            # Lambda container image definition
└── deploy.sh             # ECR build + Lambda deploy script
```

## Setup

### Prerequisites

- Python 3.12+
- AWS account with Bedrock access (Claude 3.5 Sonnet enabled in us-east-1)
- AWS credentials configured (`~/.aws/credentials` or environment variables)
- Docker (for Lambda deployment)

### Local Development

```bash
cd fraud-agent

# Install dependencies
pip install -r agent/requirements.txt -r backend/requirements.txt

# Run DynamoDB setup (uses AWS DynamoDB)
python agent/setup_dynamodb.py

# Start API server
cd backend
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

### Lambda Deployment

```bash
# Configure these variables in deploy.sh
AWS_ACCOUNT_ID=<your-account-id>
AWS_REGION=us-east-2
REPO_NAME=fraud-agent
FUNCTION_NAME=fraud-agent

# Build and deploy
chmod +x deploy.sh
./deploy.sh
```

The script:
1. Builds a Docker image from the Lambda Python 3.12 base
2. Pushes to Amazon ECR
3. Updates the Lambda function with the new image
4. Sets required environment variables (Bedrock region, DynamoDB table name)

## API

### `POST /analyze`

Analyze a transaction for fraud.

**Request body:**
```json
{
  "user_id": "user_001",
  "amount": 1200.00,
  "merchant_category": "Electronics",
  "time_of_day": "02:14 AM",
  "city": "Los Angeles, CA",
  "fraud_score_override": 0.85
}
```

`fraud_score_override` is optional — if omitted, the XGBoost classifier computes the score automatically.

**Response:**
```json
{
  "decision": "REVIEW",
  "explanation": "...",
  "fraud_score": 0.85,
  "signals": {
    "velocity": { "risk": "HIGH", "detail": "..." },
    "location": { "risk": "LOW", "detail": "..." },
    "spending": { "risk": "MEDIUM", "detail": "..." },
    "temporal": { "risk": "LOW", "detail": "..." }
  }
}
```

### `GET /`

Serves the dashboard UI.

## Evaluation

Run the evaluation suite against the live endpoint:

```bash
cd eval
pip install requests
python evaluate.py
```

Outputs `eval_results.json` with accuracy, per-class breakdown, and p95 latency.

The 40 test cases cover:
- Known users (user_001, user_002, user_003) with varied transaction patterns
- New/unknown users (no history)
- Impossible travel scenarios
- High-velocity burst transactions
- Legitimate high-value purchases

## Dashboard

The frontend provides two modes:

**Demo Scenarios** — five preset transactions with pre-configured fraud scores. Use these to explore all four decision outcomes (APPROVE, REVIEW, BLOCK, CLARIFICATION_NEEDED).

**Custom Input** — enter any transaction manually. The fraud score is computed automatically by the XGBoost classifier.

The right panel shows:
- Final decision with color-coded badge
- Fraud probability score
- Four specialist signal cards (Velocity / Location / Spending / Temporal) with HIGH / MEDIUM / LOW risk ratings
- Full LLM explanation

## Course Context

**Course**: DS5730 — Generative AI in Practice  
**Institution**: Vanderbilt University  
**Semester**: Spring 2026  

The project demonstrates:
- Supervisor multi-agent orchestration with LangGraph
- Tool-augmented LLM reasoning over structured data
- Hybrid ML + LLM decision pipeline
- Serverless deployment of an agent system on AWS Lambda
