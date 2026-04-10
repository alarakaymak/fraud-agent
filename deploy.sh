#!/bin/bash
# deploy.sh — build and deploy fraud-agent to AWS Lambda (container image)
# Prerequisites: AWS CLI configured, Docker running, ECR repo created
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh [AWS_ACCOUNT_ID] [REGION]
#
# Example:
#   ./deploy.sh 123456789012 us-east-2

set -e

ACCOUNT=${1:-$(aws sts get-caller-identity --query Account --output text)}
REGION=${2:-us-east-2}
REPO=fraud-agent
IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"
FUNCTION=fraud-agent-api

echo "==> Logging into ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Creating ECR repo (if not exists)..."
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$REPO" --region "$REGION"

echo "==> Building Docker image..."
docker build --platform linux/amd64 -t "$REPO" .
docker tag "${REPO}:latest" "$IMAGE"

echo "==> Pushing to ECR..."
docker push "$IMAGE"

echo "==> Checking if Lambda function exists..."
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" 2>/dev/null; then
  echo "==> Updating existing Lambda..."
  aws lambda update-function-code \
    --function-name "$FUNCTION" \
    --image-uri "$IMAGE" \
    --region "$REGION"
else
  echo "==> Creating Lambda function..."
  # Requires a Lambda execution role with DynamoDB + Bedrock permissions
  ROLE_ARN=$(aws iam get-role --role-name fraud-agent-lambda-role \
    --query 'Role.Arn' --output text 2>/dev/null || echo "")
  if [ -z "$ROLE_ARN" ]; then
    echo "ERROR: Create IAM role 'fraud-agent-lambda-role' with policies:"
    echo "  - AmazonDynamoDBFullAccess"
    echo "  - AmazonBedrockFullAccess"
    echo "  - AWSLambdaBasicExecutionRole"
    exit 1
  fi
  aws lambda create-function \
    --function-name "$FUNCTION" \
    --package-type Image \
    --code ImageUri="$IMAGE" \
    --role "$ROLE_ARN" \
    --timeout 120 \
    --memory-size 1024 \
    --region "$REGION" \
    --environment "Variables={
      AWS_DEFAULT_REGION=${REGION},
      DYNAMODB_HISTORY_TABLE=fraud-user-history,
      DYNAMODB_DECISIONS_TABLE=fraud-agent-decisions
    }"
fi

echo ""
echo "==> Adding Function URL (public HTTPS endpoint)..."
aws lambda add-permission \
  --function-name "$FUNCTION" \
  --statement-id public-access \
  --action lambda:InvokeFunctionUrl \
  --principal '*' \
  --function-url-auth-type NONE \
  --region "$REGION" 2>/dev/null || true

URL=$(aws lambda get-function-url-config \
  --function-name "$FUNCTION" \
  --region "$REGION" \
  --query 'FunctionUrl' --output text 2>/dev/null || \
  aws lambda create-function-url-config \
    --function-name "$FUNCTION" \
    --auth-type NONE \
    --region "$REGION" \
    --query 'FunctionUrl' --output text)

echo ""
echo "=========================================="
echo "Deployed! Function URL:"
echo "  $URL"
echo ""
echo "Update frontend/index.html:"
echo "  const API_URL = \"${URL%/}\";"
echo "=========================================="
