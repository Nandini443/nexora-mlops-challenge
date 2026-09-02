import yaml
import datetime as dt
import pathlib

def load_config(path="config.yaml"):
    """
    Load configuration from a YAML file.
    Supports environment separation, logging level, and paths.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Ensure scored_weeks.start is treated as string
    start = str(cfg["scored_weeks"]["start"])
    start_date = dt.date.fromisoformat(start)

    # Build SCORED_WEEKS dynamically
    weeks = [start_date + dt.timedelta(days=7 * i) for i in range(cfg["scored_weeks"]["count"])]
    cfg["scored_weeks"] = weeks

    # Normalize paths
    if "data_dir" in cfg:
        cfg["data_dir"] = pathlib.Path(cfg["data_dir"])
    if "output_dir" in cfg:
        cfg["output_dir"] = pathlib.Path(cfg["output_dir"])

    return cfg
