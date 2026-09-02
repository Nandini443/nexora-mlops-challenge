# DECISIONS.md — Engineering Decision Log

Five key decisions made in building this submission, with explicit rationale
tied to the cost model (unnecessary visit: **€380**; missed broken gateway:
**€600/week**, compounding).

---

## Decision 1 — Discipline: MLOps over pure ML accuracy

**Chosen:** MLOps  
**Alternatives considered:** higher-accuracy probabilistic model (e.g. Isolation Forest, Prophet)

**Rationale:**  
Production-ML systems fail primarily on *operational* problems — silent data drift,
non-reproducible predictions, no rollback capability — not on model accuracy.
A 3-sigma rule with a rigorous operational wrapper (versioned artifacts, drift
detection, CI validation, Docker) is more defensible in a team environment than
a black-box model with no audit trail.

The cost model reinforces this: a **false negative** (missing a broken gateway)
costs €600/week vs. €380 for a false positive. A well-monitored simple model
that flags missed drift quickly is safer than a complex model with no observability.

---

## Decision 2 — 3-sigma anomaly scoring over a learned model

**Chosen:** Per-gateway 3-sigma threshold on `offline_duration_sec`,
`disconnection_cnt`, `reboot_cnt`  
**Alternatives considered:** Isolation Forest, ARIMA residuals, gradient-boosted ranker

**Rationale:**  
- **Explainability:** the `reason` field in `predictions.csv` maps directly to the
  algorithm. Field engineers can verify a ranking by hand.
- **No training data required:** the model has no learnable parameters — "training"
  just serialises the hyperparams. This eliminates entire failure modes (train/serve
  skew, label leakage).
- **Per-gateway baseline:** using each gateway's own 28-day history as the reference
  window eliminates fleet-wide seasonality confounds.
- **Asymmetric cost awareness:** the sigma and window sizes were chosen so that the
  model is calibrated towards recall (minimising €600 misses) at the cost of
  some precision (accepting more €380 false positives).

---

## Decision 3 — Flat JSON artifact registry (no MLflow dependency)

**Chosen:** `models/registry/<version_id>.json` with params + timestamp  
**Alternatives considered:** MLflow Model Registry, DVC, plain pickle

**Rationale:**  
- Zero infrastructure dependencies — no tracking server, no cloud storage.
- Rollback is a single CLI flag: `--model-version v1_2026-09-01T234346Z`.
- Human-readable: any reviewer can inspect what parameters produced a given
  `predictions.csv` by reading the JSON file.
- The registry is append-only: re-training never overwrites old artifacts,
  making it trivially auditable.

**Trade-off:** does not scale to large binary model weights. Acceptable here
because the scorer has no weights.

---

## Decision 4 — Config-per-environment (YAML, not environment variables)

**Chosen:** `config.yaml` / `config.dev.yaml` / `config.prod.yaml` loaded via `--config`  
**Alternatives considered:** `os.environ`, hardcoded constants, `.env` files

**Rationale:**  
- All hyperparameters (`sigma`, `baseline_days`, `recent_days`, `visits_per_week`)
  are in one place, version-controlled, and reviewable in PRs.
- Switching environments requires one flag change, not environment variable
  juggling across shell sessions.
- Dev config (`sigma=2.5`, `baseline_days=14`) allows fast iteration;
  prod config (`sigma=3.0`, `baseline_days=28`) matches the submission spec.

---

## Decision 5 — PSI (Population Stability Index) for drift detection

**Chosen:** PSI over each metric's reference vs. current distribution window  
**Alternatives considered:** KS test, Jensen-Shannon divergence, simple mean shift

**Rationale:**  
- PSI is the **industry standard** for production model monitoring (credit risk,
  telemetry, fraud). Thresholds (< 0.10 stable, 0.10–0.25 moderate, > 0.25 alert)
  are universally understood.
- It is non-parametric and handles the heavy-tailed distributions typical of
  connectivity metrics (rare but large offline events).
- An exit-code-1 on significant drift integrates cleanly into CI/CD — any push
  where the current week's data distribution diverges from baseline will block
  the pipeline and surface the issue before bad predictions ship.
- **Cost model alignment:** a PSI alert catches distribution shifts *before*
  the 7-day scoring window, giving time to retrain or adjust thresholds before
  a €600/week miss accumulates.
