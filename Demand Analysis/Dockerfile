FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OpenMP runtime required by XGBoost (lightweight, ~140 kB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install data science dependencies, then install xgboost without heavy multi-GPU CUDA NCCL blobs
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps xgboost

COPY app/ /app/

RUN mkdir -p /app/output

CMD ["python", "predict.py"]
