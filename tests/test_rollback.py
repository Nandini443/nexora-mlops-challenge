"""test_rollback.py — Prove the model registry supports safe rollback.

MLOps property: if a new version (v2) is deployed and causes issues, an
operator can pin --model-version to an older version_id and get exactly
the same predictions that version produced originally.

Decision 3 in DECISIONS.md: we chose a flat JSON registry (no mlflow
dependency) that supports rollback via a single CLI flag.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys

import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from predict import build_predictions, load_scorer  # noqa: E402
from scoring import BaselineScorer  # noqa: E402

MONDAY = dt.date(2026, 2, 2)
SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]

V1_ID = "v1_2026-01-01T000000Z"
V2_ID = "v1_2026-02-01T000000Z"


# ---------------------------------------------------------------------------
# Test 1: Default load returns the LATEST artifact
# ---------------------------------------------------------------------------

class TestRegistryLatest:
    def test_latest_artifact_is_v2(self, registry_dir: pathlib.Path):
        """load_scorer with version_id=None must pick the lexicographically last file."""
        _, version_used = load_scorer(registry_dir, version_id=None)
        assert version_used == V2_ID, (
            f"Expected latest version to be {V2_ID!r}, got {version_used!r}"
        )

    def test_v2_params_are_loaded(self, registry_dir: pathlib.Path):
        scorer, _ = load_scorer(registry_dir, version_id=None)
        assert scorer.sigma == 2.5
        assert scorer.baseline_days == 21


# ---------------------------------------------------------------------------
# Test 2: Pinning a version_id loads the correct artifact
# ---------------------------------------------------------------------------

class TestRegistryPinnedVersion:
    def test_pin_to_v1(self, registry_dir: pathlib.Path):
        scorer, version_used = load_scorer(registry_dir, version_id=V1_ID)
        assert version_used == V1_ID
        assert scorer.sigma == 3.0
        assert scorer.baseline_days == 28

    def test_pin_to_v2(self, registry_dir: pathlib.Path):
        scorer, version_used = load_scorer(registry_dir, version_id=V2_ID)
        assert version_used == V2_ID
        assert scorer.sigma == 2.5

    def test_missing_version_raises(self, registry_dir: pathlib.Path):
        with pytest.raises(SystemExit):
            load_scorer(registry_dir, version_id="v1_9999-99-99T000000Z")


# ---------------------------------------------------------------------------
# Test 3: v1 and v2 produce DIFFERENT predictions (proves versioning matters)
# ---------------------------------------------------------------------------

class TestVersionsDifferInOutput:
    def test_v1_v2_predictions_differ(
        self, registry_dir: pathlib.Path, synthetic_telemetry: pd.DataFrame
    ):
        """Different sigma/baseline params must produce different ranked lists."""
        scorer_v1, _ = load_scorer(registry_dir, version_id=V1_ID)
        scorer_v2, _ = load_scorer(registry_dir, version_id=V2_ID)

        preds_v1 = build_predictions(scorer_v1, synthetic_telemetry, SCORED_WEEKS, 15)
        preds_v2 = build_predictions(scorer_v2, synthetic_telemetry, SCORED_WEEKS, 15)

        # They MUST differ because sigma and baseline_days are different
        assert not preds_v1.equals(preds_v2), (
            "v1 and v2 produced identical predictions — versioning has no effect"
        )


# ---------------------------------------------------------------------------
# Test 4: Rollback reproducibility — v1 output is stable across calls
# ---------------------------------------------------------------------------

class TestRollbackReproducibility:
    def test_v1_output_is_identical_on_two_calls(
        self, registry_dir: pathlib.Path, synthetic_telemetry: pd.DataFrame
    ):
        """Rollback guarantee: pinning v1 always produces the same predictions."""
        scorer_v1, _ = load_scorer(registry_dir, version_id=V1_ID)

        run_a = build_predictions(scorer_v1, synthetic_telemetry, SCORED_WEEKS, 15)
        run_b = build_predictions(scorer_v1, synthetic_telemetry, SCORED_WEEKS, 15)

        pd.testing.assert_frame_equal(run_a, run_b)

    def test_v1_after_v2_deployment_unchanged(
        self, registry_dir: pathlib.Path, synthetic_telemetry: pd.DataFrame
    ):
        """Deploy v2, then roll back to v1 — output must match original v1 run."""
        scorer_v1, _ = load_scorer(registry_dir, version_id=V1_ID)
        scorer_v2, _ = load_scorer(registry_dir, version_id=V2_ID)

        # Baseline: v1 output before v2 ever ran
        baseline_v1 = build_predictions(scorer_v1, synthetic_telemetry, SCORED_WEEKS, 15)

        # Simulate: run v2 (production deployment)
        _ = build_predictions(scorer_v2, synthetic_telemetry, SCORED_WEEKS, 15)

        # Rollback: run v1 again
        rollback_v1 = build_predictions(scorer_v1, synthetic_telemetry, SCORED_WEEKS, 15)

        pd.testing.assert_frame_equal(baseline_v1, rollback_v1)


# ---------------------------------------------------------------------------
# Test 5: Empty registry raises a clean error
# ---------------------------------------------------------------------------

class TestEmptyRegistry:
    def test_empty_registry_raises(self, tmp_path: pathlib.Path):
        empty_registry = tmp_path / "empty_registry"
        empty_registry.mkdir()
        with pytest.raises(SystemExit, match="no versioned artifacts found"):
            load_scorer(empty_registry, version_id=None)
