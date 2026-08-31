"""Save a versioned scoring artifact. For the 3-sigma baseline, 'training' just
means recording the params and metadata — there are no weights to fit."""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

from scoring import BaselineScorer


def save_version(scorer: BaselineScorer, registry_dir: pathlib.Path) -> pathlib.Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    version_id = f"{scorer.VERSION}_{timestamp}"

    metadata = {
        "version_id": version_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": scorer.params(),
    }

    out_path = registry_dir / f"{version_id}.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--baseline-days", type=int, default=28)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument(
        "--registry", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "models" / "registry",
    )
    args = parser.parse_args(argv)

    scorer = BaselineScorer(
        sigma=args.sigma, baseline_days=args.baseline_days, recent_days=args.recent_days
    )
    out_path = save_version(scorer, args.registry)
    print(f"saved version artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())