"""
AWS Lambda handler — wraps the FastAPI app via Mangum.
Deploy this with the full project directory as a Lambda zip or container image.

Environment variables required:
  AWS_DEFAULT_REGION         e.g. us-east-2
  DYNAMODB_HISTORY_TABLE     fraud-user-history
  DYNAMODB_DECISIONS_TABLE   fraud-agent-decisions
  LANGCHAIN_TRACING_V2       true               (optional, for LangSmith)
  LANGCHAIN_API_KEY          <your key>         (optional, for LangSmith)
  LANGCHAIN_PROJECT          fraud-agent        (optional, for LangSmith)
"""

from mangum import Mangum
from main import app

handler = Mangum(app, lifespan="off")
