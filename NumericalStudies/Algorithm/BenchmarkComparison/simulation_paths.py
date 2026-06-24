from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = (ROOT_DIR / "Data" / "BenchmarkComparison").resolve()
FIGURE_DIR = (ROOT_DIR / "Figure").resolve()
CONFIG_FILE = SCRIPT_DIR / "benchmark_config.json"


def load_config(config_file: str | Path = CONFIG_FILE) -> dict[str, Any]:
    path = Path(config_file)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_scenario(scenario: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(scenario))
    return clean or "unknown"


def scenario_tag(config_or_scenario: dict[str, Any] | str) -> str:
    if isinstance(config_or_scenario, dict):
        scenario = config_or_scenario.get("scenario", "unknown")
    else:
        scenario = config_or_scenario
    return f"scenario_{sanitize_scenario(str(scenario).strip().lower())}"


def scenario_data_dir(
    config_or_scenario: dict[str, Any] | str,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    run_dir = os.environ.get("SIMULATION_RUN_DIR")
    base_dir = Path(run_dir) if run_dir else Path(data_dir)
    if not base_dir.is_absolute():
        base_dir = (SCRIPT_DIR / base_dir).resolve()
    return base_dir / scenario_tag(config_or_scenario)


def scenario_csv_path(
    filename: str | Path,
    config_or_scenario: dict[str, Any] | str,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    path = Path(filename)
    tag = scenario_tag(config_or_scenario)
    tagged_name = f"{path.stem}_{tag}{path.suffix}"
    return scenario_data_dir(config_or_scenario, data_dir) / tagged_name
