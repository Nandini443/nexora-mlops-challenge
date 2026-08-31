"""Load a versioned scorer artifact, score all required weeks, write predictions.csv."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import pandas as pd

from scoring import BaselineScorer

SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]
VISITS_PER_WEEK = 15
MAX_REASON_CHARS = 300


def load_telemetry(data_dir: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        data_dir / "telemetry",
        columns=["gateway_id", "ts_utc", "offline_duration_sec", "disconnection_cnt", "reboot_cnt"],
    )
    frame["ts"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame.drop(columns=["ts_utc"])


def load_scorer(registry_dir: pathlib.Path, version_id: str | None) -> tuple[BaselineScorer, str]:
    if version_id is None:
        # pick the most recently created artifact
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


def build_predictions(scorer: BaselineScorer, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for monday in SCORED_WEEKS:
        ranked = scorer.score_week(frame, monday, visits_per_week=VISITS_PER_WEEK)
        if len(ranked) < VISITS_PER_WEEK:
            raise SystemExit(f"only {len(ranked)} gateways have data before {monday}")
        for rank, row in enumerate(ranked.head(VISITS_PER_WEEK).itertuples(index=False), 1):
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
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=repo_root / "data")
    parser.add_argument("--registry", type=pathlib.Path, default=repo_root / "models" / "registry")
    parser.add_argument("--model-version", type=str, default=None,
                         help="specific version_id to load; defaults to most recent")
    parser.add_argument("--out", type=pathlib.Path, default=repo_root / "predictions.csv")
    args = parser.parse_args(argv)

    scorer, version_used = load_scorer(args.registry, args.model_version)
    frame = load_telemetry(args.data)
    predictions = build_predictions(scorer, frame)
    predictions.to_csv(args.out, index=False)

    print(f"used model version: {version_used}")
    print(f"wrote {args.out} — {len(predictions)} rows over {predictions.week_start.nunique()} weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())