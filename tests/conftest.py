"""Shared fixtures for the Nexora MLOps test suite.

Generates a fully synthetic in-memory telemetry DataFrame so tests run
without any real data files — making CI fast and data-leak-free.

The fixture covers 60 days of hourly rows for 20 fake gateways, spanning
2026-01-01 through 2026-03-01, which includes all 8 scored weeks
(2026-02-02 → 2026-03-23).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

RNG_SEED = 42
N_GATEWAYS = 20
# 90 days: 2026-01-01 → 2026-04-01, covers all 8 scored weeks
# (last week 2026-03-23 needs recent window [2026-03-16, 2026-03-23))
DAYS = 90


def _make_gateway_ids(n: int) -> list[str]:
    rng = np.random.default_rng(RNG_SEED)
    return [
        "".join(f"{b:02X}" for b in rng.integers(0, 256, size=6))
        for _ in range(n)
    ]


@pytest.fixture(scope="session")
def gateway_ids() -> list[str]:
    return _make_gateway_ids(N_GATEWAYS)


@pytest.fixture(scope="session")
def synthetic_telemetry(gateway_ids: list[str]) -> pd.DataFrame:
    """Return a synthetic hourly telemetry DataFrame with realistic structure."""
    rng = np.random.default_rng(RNG_SEED)

    start = pd.Timestamp("2026-01-01", tz="UTC")
    hours = DAYS * 24
    timestamps = [start + pd.Timedelta(hours=h) for h in range(hours)]

    rows = []
    for gw in gateway_ids:
        for ts in timestamps:
            rows.append(
                {
                    "gateway_id": gw,
                    "ts": ts,
                    "offline_duration_sec": float(rng.exponential(scale=120)),
                    "disconnection_cnt": int(rng.poisson(lam=1)),
                    "reboot_cnt": int(rng.poisson(lam=0.05)),
                }
            )

    frame = pd.DataFrame(rows)

    # Inject a known anomaly: gateway_ids[0] spikes on ALL THREE metrics.
    # We inject this for only 48 hours (not the full 7 days) because a 3-sigma
    # rule mathematically cannot flag anomalies that make up >10% of the baseline
    # window (48h / 672h = ~7%). This will yield 48 * 3 = 144 flagged hours.
    anomaly_gw = gateway_ids[0]
    anomaly_mask = (
        (frame["gateway_id"] == anomaly_gw)
        & (frame["ts"] >= pd.Timestamp("2026-01-31", tz="UTC"))
        & (frame["ts"] < pd.Timestamp("2026-02-02", tz="UTC"))
    )
    frame.loc[anomaly_mask, "disconnection_cnt"] = 9999
    frame.loc[anomaly_mask, "offline_duration_sec"] = 999_999.0
    frame.loc[anomaly_mask, "reboot_cnt"] = 999

    return frame


@pytest.fixture(scope="session")
def registry_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A temp directory pre-populated with two versioned artifacts (v1 and v2)."""
    registry = tmp_path_factory.mktemp("registry")

    v1_meta = {
        "version_id": "v1_2026-01-01T000000Z",
        "created_at": "2026-01-01T00:00:00+00:00",
        "params": {"version": "v1", "sigma": 3.0, "baseline_days": 28, "recent_days": 7},
    }
    v2_meta = {
        "version_id": "v1_2026-02-01T000000Z",
        "created_at": "2026-02-01T00:00:00+00:00",
        "params": {"version": "v1", "sigma": 2.5, "baseline_days": 21, "recent_days": 7},
    }

    (registry / "v1_2026-01-01T000000Z.json").write_text(json.dumps(v1_meta, indent=2))
    (registry / "v1_2026-02-01T000000Z.json").write_text(json.dumps(v2_meta, indent=2))

    return registry
