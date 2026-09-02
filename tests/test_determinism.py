"""test_determinism.py — Prove the pipeline is reproducible.

MLOps property: given identical inputs and a fixed version artifact, every
run of score_week() and build_predictions() must produce bit-for-bit identical
output. Any non-determinism (e.g. from random seeds, timestamp-dependent
sorting, or floating-point hash ordering) would make audit trails impossible.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import pathlib

import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from scoring import BaselineScorer  # noqa: E402
from predict import build_predictions  # noqa: E402


MONDAY = dt.date(2026, 2, 2)
SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]


def _df_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame's canonical CSV representation."""
    csv_bytes = df.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Test 1: score_week() is deterministic
# ---------------------------------------------------------------------------

class TestScoreWeekDeterminism:
    def test_same_result_on_repeated_call(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)

        result_a = scorer.score_week(synthetic_telemetry, MONDAY)
        result_b = scorer.score_week(synthetic_telemetry, MONDAY)

        pd.testing.assert_frame_equal(result_a, result_b)

    def test_hash_is_stable(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)

        h1 = _df_hash(scorer.score_week(synthetic_telemetry, MONDAY))
        h2 = _df_hash(scorer.score_week(synthetic_telemetry, MONDAY))

        assert h1 == h2, "score_week() hash changed between calls — non-determinism detected"

    def test_known_anomaly_gateway_is_top_ranked(
        self, synthetic_telemetry: pd.DataFrame, gateway_ids: list[str]
    ):
        """Injected anomaly in conftest must surface as rank-1 for week 2026-02-02."""
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        ranked = scorer.score_week(synthetic_telemetry, MONDAY)
        assert ranked.iloc[0]["gateway_id"] == gateway_ids[0], (
            "Expected the injected-anomaly gateway to be rank 1"
        )


# ---------------------------------------------------------------------------
# Test 2: build_predictions() full-pipeline determinism
# ---------------------------------------------------------------------------

class TestBuildPredictionsDeterminism:
    def test_full_pipeline_identical_output(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)

        preds_a = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)
        preds_b = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)

        pd.testing.assert_frame_equal(preds_a, preds_b)

    def test_row_count(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        preds = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)
        assert len(preds) == 8 * 15, f"Expected 120 rows, got {len(preds)}"

    def test_week_count(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        preds = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)
        assert preds["week_start"].nunique() == 8

    def test_ranks_are_sequential_per_week(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        preds = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)

        for week, group in preds.groupby("week_start"):
            ranks = sorted(group["rank"].tolist())
            assert ranks == list(range(1, 16)), (
                f"Week {week}: expected ranks 1-15, got {ranks}"
            )

    def test_no_duplicate_gateways_within_week(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        preds = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)

        for week, group in preds.groupby("week_start"):
            ids = group["gateway_id"].tolist()
            assert len(ids) == len(set(ids)), (
                f"Week {week}: duplicate gateway_id in predictions"
            )

    def test_reason_within_char_limit(self, synthetic_telemetry: pd.DataFrame):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        preds = build_predictions(scorer, synthetic_telemetry, SCORED_WEEKS, 15)
        too_long = (preds["reason"].str.len() > 300).sum()
        assert too_long == 0, f"{too_long} reason(s) exceed 300 chars"


# ---------------------------------------------------------------------------
# Test 3: Artifact params round-trip
# ---------------------------------------------------------------------------

class TestArtifactParamsRoundTrip:
    def test_params_are_reproducible(self):
        scorer = BaselineScorer(sigma=2.5, baseline_days=21, recent_days=5)
        params = scorer.params()

        # Reconstruct scorer from params (as predict.py does)
        reconstructed = BaselineScorer(
            sigma=params["sigma"],
            baseline_days=params["baseline_days"],
            recent_days=params["recent_days"],
        )
        assert reconstructed.sigma == scorer.sigma
        assert reconstructed.baseline_days == scorer.baseline_days
        assert reconstructed.recent_days == scorer.recent_days

    def test_params_json_serialisable(self):
        scorer = BaselineScorer(sigma=3.0, baseline_days=28, recent_days=7)
        # Must not raise
        json.dumps(scorer.params())
