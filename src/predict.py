"""Load a versioned scorer artifact, score all required weeks, write predictions.csv."""

from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import pathlib

import pandas as pd

from scoring import BaselineScorer
from config import load_config

MAX_REASON_CHARS = 300


def load_telemetry(data_dir: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        data_dir / "telemetry",
        columns=["gateway_id", "ts_utc", "offline_duration_sec", "disconnection_cnt", "reboot_cnt"],
    )
    frame["ts"] = pd.to_datetime(frame["ts_utc"], utc=True)
    logging.info("Loaded telemetry data with %d rows", len(frame))
    return frame.drop(columns=["ts_utc"])


def load_scorer(registry_dir: pathlib.Path, version_id: str | None) -> tuple[BaselineScorer, str]:
    if version_id is None:
        candidates = sorted(registry_dir.glob("*.json"))
        if not candidates:
            raise SystemExit(f"no versioned artifacts found in {registry_dir}; run train.py first")
        artifact_path = candidates[-1]
    else:
        artifact_path = registry_dir / f"{version_id}.json"
        if not artifact_path.exists():
            raise SystemExit(f"no artifact found at {artifact_path}")

    metadata = json.loads(artifact_path.read_text())
    params = metadata["params"]
    scorer = BaselineScorer(
        sigma=params["sigma"],
        baseline_days=params["baseline_days"],
        recent_days=params["recent_days"],
    )
    return scorer, metadata["version_id"]


def build_predictions(
    scorer: BaselineScorer,
    frame: pd.DataFrame,
    scored_weeks: list[dt.date],
    visits_per_week: int,
) -> pd.DataFrame:
    rows = []
    for monday in scored_weeks:
        ranked = scorer.score_week(frame, monday, visits_per_week=visits_per_week)
        if len(ranked) < visits_per_week:
            raise SystemExit(f"only {len(ranked)} gateways have data before {monday}")
        for rank, row in enumerate(ranked.head(visits_per_week).itertuples(index=False), 1):
            metric = row.worst_metric or "no metric over 3 sigma"
            reason = (
                f"{row.flagged_hours} hour(s) beyond {scorer.sigma} sigma of this gateway's "
                f"own {scorer.baseline_days}-day baseline in the last {scorer.recent_days} days; "
                f"first breach on {metric}"
            )[:MAX_REASON_CHARS]
            rows.append(
                {
                    "week_start": monday.isoformat(),
                    "rank": rank,
                    "gateway_id": row.gateway_id,
                    "score": float(row.flagged_hours),
                    "reason": reason,
                }
            )
    logging.info("Built predictions for %d weeks", len(scored_weeks))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (e.g., config.dev.yaml)")
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--registry", type=pathlib.Path, default=pathlib.Path("models/registry"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions.csv"))
    parser.add_argument("--model-version", type=str, default=None,
                        help="Specific version_id to load (default: latest)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.get("logging_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scored_weeks: list[dt.date] = cfg["scored_weeks"]
    visits_per_week: int = cfg["visits_per_week"]

    logging.info("Loading scorer from registry: %s", args.registry)
    scorer, version_used = load_scorer(args.registry, args.model_version)
    logging.info("Using model version: %s", version_used)

    frame = load_telemetry(args.data)
    predictions = build_predictions(scorer, frame, scored_weeks, visits_per_week)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)

    logging.info("Predictions written to %s with %d rows", args.out, len(predictions))
    print(f"used model version: {version_used}")
    print(f"wrote {args.out} — {len(predictions)} rows over {predictions.week_start.nunique()} weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
