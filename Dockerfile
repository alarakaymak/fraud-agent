# Lambda container image — fraud detection agent
# Build: docker build -t fraud-agent .
# Deploy: see deploy.sh

FROM public.ecr.aws/lambda/python:3.12

# Install system deps (XGBoost needs libgomp)
RUN dnf install -y libgomp && dnf clean all

# Install Python deps
COPY agent/requirements.txt      /tmp/agent-req.txt
COPY backend/requirements.txt    /tmp/backend-req.txt
RUN pip install -r /tmp/agent-req.txt -r /tmp/backend-req.txt --no-cache-dir

# Copy source
COPY agent/       ${LAMBDA_TASK_ROOT}/agent/
COPY backend/     ${LAMBDA_TASK_ROOT}/backend/
COPY classifier/  ${LAMBDA_TASK_ROOT}/classifier/
COPY frontend/    ${LAMBDA_TASK_ROOT}/frontend/

# Lambda Python path
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/backend:${LAMBDA_TASK_ROOT}/agent:${LAMBDA_TASK_ROOT}"

CMD ["backend.lambda_function.handler"]
