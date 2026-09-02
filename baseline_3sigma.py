#!/usr/bin/env python3
from __future__ import annotations
"""A 3-sigma anomaly baseline. THIS SHIPS TO STUDENTS."""

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import argparse
import datetime as dt
import pathlib

import numpy as np
import pandas as pd
from src.config import load_config


cfg = load_config()
METRICS = ["offline_duration_sec", "disconnection_cnt", "reboot_cnt"]
SCORED_WEEKS = cfg["scored_weeks"]
VISITS_PER_WEEK = cfg["visits_per_week"]
BASELINE_DAYS = cfg["baseline_days"]
RECENT_DAYS = cfg["recent_days"]
SIGMA = cfg["sigma"]


def load(data_dir: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        data_dir / "telemetry", columns=["gateway_id", "ts_utc", *METRICS]
    )
    frame["ts"] = pd.to_datetime(frame["ts_utc"], utc=True)
    logging.info("Loaded telemetry data with %d rows", len(frame))
    return frame.drop(columns=["ts_utc"])


def rank_week(frame: pd.DataFrame, monday: dt.date) -> pd.DataFrame:
    logging.info("Ranking anomalies for week starting %s", monday.isoformat())
    end = pd.Timestamp(monday, tz="UTC")
    window = frame[(frame["ts"] >= end - dt.timedelta(days=BASELINE_DAYS)) & (frame["ts"] < end)]
    if window.empty:
        logging.warning("No data available for week starting %s", monday.isoformat())
        return pd.DataFrame(columns=["gateway_id", "flagged_hours", "worst_metric"])

    stats = window.groupby("gateway_id")[METRICS].agg(["mean", "std"])
    recent = window[window["ts"] >= end - dt.timedelta(days=RECENT_DAYS)].copy()

    flags = pd.Series(0, index=recent.index, dtype=int)
    worst = pd.Series("", index=recent.index, dtype=object)
    for metric in METRICS:
        mean = recent["gateway_id"].map(stats[(metric, "mean")])
        std = recent["gateway_id"].map(stats[(metric, "std")]).replace(0, np.nan)
        exceeded = (recent[metric] - mean) > SIGMA * std
        exceeded = exceeded.fillna(False)
        flags = flags + exceeded.astype(int)
        worst = worst.where(~exceeded | (worst != ""), metric)

    recent["flagged"] = flags
    recent["worst_metric"] = worst
    grouped = recent.groupby("gateway_id").agg(
        flagged_hours=("flagged", "sum"),
        worst_metric=("worst_metric", lambda s: next((v for v in s if v), "")),
    )
    logging.info("Ranked %d gateways for week starting %s", len(grouped), monday.isoformat())
    return grouped.sort_values("flagged_hours", ascending=False).reset_index()


def build_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building predictions for %d weeks", len(SCORED_WEEKS))
    rows = []
    for monday in SCORED_WEEKS:
        ranked = rank_week(frame, monday)
        if len(ranked) < VISITS_PER_WEEK:
            raise SystemExit(f"only {len(ranked)} gateways have data before {monday}")
        for rank, row in enumerate(ranked.head(VISITS_PER_WEEK).itertuples(index=False), 1):
            metric = row.worst_metric or "no metric over 3 sigma"
            rows.append(
                {
                    "week_start": monday.isoformat(),
                    "rank": rank,
                    "gateway_id": row.gateway_id,
                    "score": float(row.flagged_hours),
                    "reason": (
                        f"{row.flagged_hours} hour(s) beyond {SIGMA} sigma of this gateway's own "
                        f"{BASELINE_DAYS}-day baseline in the last {RECENT_DAYS} days; first breach on {metric}"
                    ),
                }
            )
    logging.info("Finished building predictions with %d rows", len(rows))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = here / "data" if (here / "data").exists() else here.parent / "student-brief" / "data"
    parser.add_argument("--data", type=pathlib.Path, default=default_data)
    parser.add_argument("--out", type=pathlib.Path, default=here / "predictions_baseline.csv")
    args = parser.parse_args(argv)

    logging.info("Starting baseline scoring run")
    frame = load(args.data)
    predictions = build_predictions(frame)
    predictions.to_csv(args.out, index=False)
    logging.info("Baseline predictions written to %s with %d rows", args.out, len(predictions))

    print(f"wrote {args.out} — {len(predictions)} rows over {predictions.week_start.nunique()} weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
