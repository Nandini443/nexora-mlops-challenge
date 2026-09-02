"""Save a versioned scoring artifact. For the 3-sigma baseline, 'training' just
means recording the params and metadata — there are no weights to fit."""

from __future__ import annotations
import argparse
import json
import logging
import pathlib
from datetime import datetime, timezone

from scoring import BaselineScorer
from config import load_config


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
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (e.g., config.dev.yaml)")
    parser.add_argument("--registry", type=pathlib.Path, default=pathlib.Path("models/registry"),
                        help="Directory to write versioned artifacts into")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--baseline-days", type=int, default=None)
    parser.add_argument("--recent-days", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.get("logging_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # CLI flags override config file values
    sigma = args.sigma if args.sigma is not None else cfg["sigma"]
    baseline_days = args.baseline_days if args.baseline_days is not None else cfg["baseline_days"]
    recent_days = args.recent_days if args.recent_days is not None else cfg["recent_days"]

    logging.info(
        "Starting training with sigma=%s, baseline_days=%s, recent_days=%s",
        sigma, baseline_days, recent_days,
    )

    scorer = BaselineScorer(sigma=sigma, baseline_days=baseline_days, recent_days=recent_days)
    out_path = save_version(scorer, args.registry)

    logging.info("Saved version artifact at %s", out_path)
    print(f"saved version artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
