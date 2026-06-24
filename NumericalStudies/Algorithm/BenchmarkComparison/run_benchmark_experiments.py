"""Run fixed-split benchmark replications with isolated, resumable outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = ROOT_DIR / "Data" / "BenchmarkComparison"
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"
METHODS_DIR = SCRIPT_DIR / "Methods"
ALGORITHMS = [
    "nv_DNN.py",
    "nv_ETO.py",
    "nv_KO.py",
    "nv_KPQR.py",
    "nv_LinearModel.py",
    "nv_LinearModel_L1.py",
    "nv_LinearModel_L2.py",
    "nv_Oracle.py",
    "nv_RKHS.py",
    "nv_SAA.py",
    "nv_scarfminimax.py",
]


def print_banner(status: str, scenario: str, seed: int, task: str):
    line = "=" * 88
    print(f"\n{line}", flush=True)
    print(
        f"{status}: scenario={scenario.upper()} | seed={seed} | task={task}",
        flush=True,
    )
    print(line, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=["a", "b"], default=["a", "b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    return parser.parse_args()


def main():
    args = parse_args()
    original_config_text = CONFIG_PATH.read_text(encoding="utf-8")
    original_config = json.loads(original_config_text)
    try:
        for scenario in args.scenarios:
            for seed in args.seeds:
                run_root = (DATA_DIR / f"scenario_{scenario}" / f"seed_{seed}").resolve()
                run_root.mkdir(parents=True, exist_ok=True)
                config = {
                    **original_config,
                    "total_len": 5000,
                    "lntr": 3000,
                    "lnva": 1000,
                    "lnte": 1000,
                    "scenario": scenario,
                    "seed": seed,
                    "tau": 0.75,
                }
                CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")
                env = {**os.environ, "SIMULATION_RUN_DIR": str(run_root)}

                data_marker = run_root / ".done_data_generation"
                if not data_marker.exists():
                    print_banner("STARTING", scenario, seed, "generate_benchmark_data.py")
                    subprocess.run(
                        [sys.executable, "generate_benchmark_data.py"],
                        cwd=SCRIPT_DIR,
                        env=env,
                        check=True,
                    )
                    data_marker.write_text("done\n", encoding="utf-8")
                    print_banner("COMPLETED", scenario, seed, "generate_benchmark_data.py")
                else:
                    print_banner("SKIPPED", scenario, seed, "generate_benchmark_data.py (already complete)")

                for script in args.algorithms:
                    marker = run_root / f".done_{Path(script).stem}"
                    if marker.exists():
                        print_banner("SKIPPED", scenario, seed, f"{script} (already complete)")
                        continue
                    print_banner("STARTING", scenario, seed, script)
                    subprocess.run([sys.executable, str(METHODS_DIR / script)], cwd=SCRIPT_DIR, env=env, check=True)
                    marker.write_text("done\n", encoding="utf-8")
                    print_banner("COMPLETED", scenario, seed, script)

                if set(args.algorithms) == set(ALGORITHMS):
                    (run_root / ".complete").write_text("done\n", encoding="utf-8")
    finally:
        CONFIG_PATH.write_text(original_config_text, encoding="utf-8")


if __name__ == "__main__":
    main()
