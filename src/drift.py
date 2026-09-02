"""Detects distribution shift between the training baseline window and the current
scoring window using Population Stability Index (PSI).

PSI interpretation:
  < 0.10  → stable        (no action needed)
  0.10–0.25 → moderate    (monitor)
  > 0.25  → significant   (retrain / investigate)

Exit codes:
  0 → all metrics stable
  1 → at least one metric shows significant drift (PSI > 0.25)
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

METRICS = ["offline_duration_sec", "disconnection_cnt", "reboot_cnt"]
N_BINS = 10
PSI_WARN = 0.10
PSI_ALERT = 0.25


# ---------------------------------------------------------------------------
# Core PSI calculation
# ---------------------------------------------------------------------------

def _psi_for_metric(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = N_BINS,
) -> float:
    """Compute PSI between two numeric series using quantile binning on reference."""
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0

    # Build bin edges from reference distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(reference, quantiles))

    # Need at least 2 unique edges to form bins
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Avoid division by zero — replace 0 counts with 0.5 (standard PSI practice)
    ref_pct = np.where(ref_counts == 0, 0.5, ref_counts) / len(reference)
    cur_pct = np.where(cur_counts == 0, 0.5, cur_counts) / len(current)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_telemetry(data_dir: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        data_dir / "telemetry",
        columns=["gateway_id", "ts_utc", *METRICS],
    )
    frame["ts"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame.drop(columns=["ts_utc"])


# ---------------------------------------------------------------------------
# Main drift check
# ---------------------------------------------------------------------------

def check_drift(
    frame: pd.DataFrame,
    reference_end: pd.Timestamp,
    reference_days: int = 28,
    current_days: int = 7,
) -> dict:
    """
    Split the telemetry into reference and current windows and compute PSI
    for each metric.

    reference window: [reference_end - reference_days, reference_end)
    current window:   [reference_end - current_days,   reference_end)
    """
    ref_start = reference_end - pd.Timedelta(days=reference_days)
    cur_start = reference_end - pd.Timedelta(days=current_days)

    reference = frame[(frame["ts"] >= ref_start) & (frame["ts"] < reference_end)]
    current = frame[(frame["ts"] >= cur_start) & (frame["ts"] < reference_end)]

    results = {}
    significant_drift = False

    for metric in METRICS:
        psi = _psi_for_metric(reference[metric], current[metric])
        if psi >= PSI_ALERT:
            status = "SIGNIFICANT"
            significant_drift = True
        elif psi >= PSI_WARN:
            status = "MODERATE"
        else:
            status = "STABLE"
        results[metric] = {"psi": psi, "status": status}
        logging.info("  %-30s PSI=%.4f  [%s]", metric, psi, status)

    return {
        "reference_window": {
            "start": ref_start.isoformat(),
            "end": reference_end.isoformat(),
            "rows": len(reference),
        },
        "current_window": {
            "start": cur_start.isoformat(),
            "end": reference_end.isoformat(),
            "rows": len(current),
        },
        "metrics": results,
        "significant_drift_detected": significant_drift,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"),
                        help="Directory containing the telemetry/ parquet partition")
    parser.add_argument("--reference-days", type=int, default=28,
                        help="Size of the reference (baseline) window in days")
    parser.add_argument("--current-days", type=int, default=7,
                        help="Size of the current (recent) window in days")
    parser.add_argument("--as-of", type=str, default=None,
                        help="ISO date to treat as 'now' (default: max ts in data)")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Write JSON report here (default: models/drift_report_<ts>.json)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Loading telemetry from %s", args.data)
    frame = load_telemetry(args.data)

    if args.as_of:
        reference_end = pd.Timestamp(args.as_of, tz="UTC")
    else:
        reference_end = frame["ts"].max()
    logging.info("Reference end (as-of): %s", reference_end)

    logging.info("Computing PSI drift scores (reference=%dd, current=%dd)...",
                 args.reference_days, args.current_days)
    report = check_drift(
        frame,
        reference_end=reference_end,
        reference_days=args.reference_days,
        current_days=args.current_days,
    )

    # Write report
    if args.out is None:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = pathlib.Path("models") / f"drift_report_{ts_str}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logging.info("Drift report written to %s", args.out)

    # Summary
    any_moderate = any(
        m["status"] != "STABLE" for m in report["metrics"].values()
    )
    if report["significant_drift_detected"]:
        print(f"[ALERT] Significant drift detected — see {args.out}")
        return 1
    elif any_moderate:
        print(f"[WARN]  Moderate drift detected — monitor — see {args.out}")
        return 0
    else:
        print(f"[OK]    All metrics stable — see {args.out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
