# Nexora 2026 — MLOps Track Submission

> **Task:** Given telemetry data up to a Monday, output the 15 gateways to
> visit that week, ranked, with reasons. 8 weeks: Feb 2 – Mar 23, 2026.
> 120 rows total in `predictions.csv`.

---

## Cost model

| Event | Cost |
|---|---|
| Unnecessary visit (false positive) | €380 one-time |
| Missed broken gateway (false negative) | **€600/week**, compounding |

The model is calibrated to minimise misses (recall-first) at the cost of some
unnecessary visits.

---

## Quick Start

```bash
git clone https://github.com/Nandini443/nexora-mlops-challenge
cd nexora-mlops-challenge
pip install -r requirements.txt
```

### 1. Train

Saves a versioned JSON artifact to `models/registry/`:

```bash
python src/train.py --registry models/registry
# or with a custom config:
python src/train.py --registry models/registry --config config.prod.yaml
```

### 2. Predict

Loads the latest artifact and writes `predictions.csv`:

```bash
python src/predict.py --data data --registry models/registry --out predictions.csv
```

To pin a specific version (rollback):

```bash
python src/predict.py \
  --data data \
  --registry models/registry \
  --out predictions.csv \
  --model-version v1_2026-09-01T234346Z
```

### 3. Validate

```bash
python validate_submission.py predictions.csv
# predictions.csv: OK
#   15 ranked gateways for each of 8 weeks, 2026-02-02 to 2026-03-23
```

### 4. Run everything at once

```bash
# Makefile
make all

# PowerShell
.\run.ps1

# Bash
bash run.sh
```

---

## Docker

```bash
# Build image
docker build -t nexora-mlops .

# Run individual services
docker compose run train
docker compose run predict
docker compose run validate
docker compose run drift
docker compose run test        # pytest — no real data needed
```

---

## Run Tests

```bash
pytest tests/ -v
```

Tests use **synthetic in-memory data only** — no real telemetry files needed.

| Test file | What it proves |
|---|---|
| `tests/test_determinism.py` | Identical inputs → identical outputs (bit-for-bit) |
| `tests/test_rollback.py` | v2 deploy → v1 rollback gives original predictions |

---

## Drift Detection

```bash
python src/drift.py --data data --registry models/registry
# [OK]    All metrics stable — see models/drift_report_20260902T....json
# exit 0 = stable, exit 1 = significant drift (CI will catch this)
```

PSI thresholds: `< 0.10` stable · `0.10–0.25` moderate · `> 0.25` alert.

---

## Rollback

```bash
# 1. List available versions
ls models/registry/

# 2. Re-run predict pinned to an older version
python src/predict.py \
  --data data \
  --registry models/registry \
  --out predictions_rollback.csv \
  --model-version v1_2026-09-01T131726Z

# 3. Validate rollback output
python validate_submission.py predictions_rollback.csv
```

---

## Environment Configs

| File | Purpose |
|---|---|
| `config.yaml` | Default (prod params, 8 weeks) |
| `config.prod.yaml` | Production: σ=3.0, baseline=28d, recent=7d |
| `config.dev.yaml` | Development: σ=2.5, baseline=14d, 4 weeks |

---

## File Map

```
nexora-mlops-challenge/
├── src/
│   ├── scoring.py          # BaselineScorer class (3-sigma, versioned)
│   ├── train.py            # Save versioned artifact to models/registry/
│   ├── predict.py          # Load artifact → write predictions.csv
│   ├── drift.py            # PSI drift detection
│   └── config.py           # YAML config loader
├── tests/
│   ├── conftest.py         # Synthetic telemetry fixture (no real data)
│   ├── test_determinism.py # Reproducibility tests
│   └── test_rollback.py    # Version rollback tests
├── models/
│   └── registry/           # Versioned JSON artifacts (gitignored data, not models)
├── data/                   # Telemetry parquet (gitignored)
├── .github/workflows/
│   └── pipeline.yaml       # CI: test → validate
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config.yaml
├── config.prod.yaml
├── config.dev.yaml
├── validate_submission.py  # Schema checker (ships to students)
├── baseline_3sigma.py      # Standalone baseline (ships to students)
├── predictions.csv         # 120-row submission output
├── DECISIONS.md            # 5 engineering decisions with rationale
├── AI-USAGE.md             # AI assistance disclosure
└── makefile                # make train / predict / validate / all
```

---

## CI/CD

GitHub Actions runs on every push:
1. **`test` job** — `pytest tests/ -v` (synthetic data, fast)
2. **`pipeline` job** — validates committed `predictions.csv`, smoke-tests `train.py`

Green badge = schema is valid and the pipeline is importable.

---

## Engineering Decisions

See [DECISIONS.md](DECISIONS.md) for the full decision log. Summary:

1. **Discipline: MLOps** — production rigor over model accuracy
2. **3-sigma scorer** — explainable, auditable, no training data required
3. **Flat JSON registry** — rollback with one flag, no infrastructure
4. **YAML config-per-environment** — all hyperparams in version control
5. **PSI drift detection** — industry-standard, CI-integrable, cost-model-aligned

project video link:
https://drive.google.com/file/d/1JYTiANLAS8ptePc-S4S0FX-NHLr84U-J/view?usp=sharing
