import csv
import json
import os
from datetime import datetime

from simulation_paths import scenario_csv_path


def _resolve_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path_str))


def record_runtime(
    model_name,
    total_test_runtime_sec,
    test_samples,
    config_file = "benchmark_config.json",
    output_file="../Data/runtime_metrics.csv",
):
    total_runtime = float(total_test_runtime_sec)
    sample_count = int(test_samples)
    per_sample_runtime = total_runtime / sample_count if sample_count > 0 else float("nan")

    scenario = "unknown"
    config_path = _resolve_path(config_file)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            scenario = str(config.get("scenario", "unknown"))
        except Exception:
            scenario = "unknown"

    if output_file == "../Data/runtime_metrics.csv":
        output_path = str(scenario_csv_path("runtime_metrics.csv", scenario))
    else:
        output_path = _resolve_path(output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "model_name",
        "scenario",
        "test_samples",
        "total_test_runtime_sec",
        "per_sample_runtime_sec",
    ]
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "scenario": scenario,
        "test_samples": sample_count,
        "total_test_runtime_sec": total_runtime,
        "per_sample_runtime_sec": per_sample_runtime,
    }

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        "Runtime logged -> "
        f"{output_path} | {model_name}: total={total_runtime:.6f}s, per_sample={per_sample_runtime:.6f}s"
    )
