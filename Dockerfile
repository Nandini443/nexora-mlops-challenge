# ── Stage 1: dependency layer (cached unless requirements.txt changes) ────────
FROM python:3.10-slim AS deps

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: application ──────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app
COPY src/       ./src/
COPY config.yaml config.dev.yaml config.prod.yaml ./
COPY validate_submission.py baseline_3sigma.py ./
COPY models/    ./models/

# data/ is gitignored — mount it at runtime with -v ./data:/app/data
# predictions.csv is written to /app at runtime

ENTRYPOINT ["python"]
CMD ["src/predict.py", \
     "--data",     "data", \
     "--registry", "models/registry", \
     "--out",      "predictions.csv"]
