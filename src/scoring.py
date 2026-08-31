"""Gateway anomaly scoring — refactored baseline as a versioned, reusable class."""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

METRICS = ["offline_duration_sec", "disconnection_cnt", "reboot_cnt"]


class BaselineScorer:
    """3-sigma anomaly scorer. Wraps the original baseline_3sigma.py logic
    so it can be versioned, saved, and loaded like any other model artifact."""

    VERSION = "v1"

    def __init__(self, sigma: float = 3.0, baseline_days: int = 28, recent_days: int = 7):
        self.sigma = sigma
        self.baseline_days = baseline_days
        self.recent_days = recent_days

    def params(self) -> dict:
        """Everything needed to reproduce this scorer's behavior exactly."""
        return {
            "version": self.VERSION,
            "sigma": self.sigma,
            "baseline_days": self.baseline_days,
            "recent_days": self.recent_days,
        }

    def score_week(self, frame: pd.DataFrame, monday: dt.date, visits_per_week: int = 15) -> pd.DataFrame:
        """Rank gateways for the week starting `monday`. Returns columns:
        gateway_id, flagged_hours, worst_metric — sorted worst first."""
        end = pd.Timestamp(monday, tz="UTC")
        window = frame[
            (frame["ts"] >= end - dt.timedelta(days=self.baseline_days))
            & (frame["ts"] < end)
        ]
        if window.empty:
            return pd.DataFrame(columns=["gateway_id", "flagged_hours", "worst_metric"])

        stats = window.groupby("gateway_id")[METRICS].agg(["mean", "std"])
        recent = window[window["ts"] >= end - dt.timedelta(days=self.recent_days)].copy()

        flags = pd.Series(0, index=recent.index, dtype=int)
        worst = pd.Series("", index=recent.index, dtype=object)
        for metric in METRICS:
            mean = recent["gateway_id"].map(stats[(metric, "mean")])
            std = recent["gateway_id"].map(stats[(metric, "std")]).replace(0, np.nan)
            exceeded = ((recent[metric] - mean) > self.sigma * std).fillna(False)
            flags += exceeded.astype(int)
            worst = worst.where(~exceeded | (worst != ""), metric)

        recent["flagged"] = flags
        recent["worst_metric"] = worst
        grouped = recent.groupby("gateway_id").agg(
            flagged_hours=("flagged", "sum"),
            worst_metric=("worst_metric", lambda s: next((v for v in s if v), "")),
        )
        return grouped.sort_values("flagged_hours", ascending=False).reset_index()