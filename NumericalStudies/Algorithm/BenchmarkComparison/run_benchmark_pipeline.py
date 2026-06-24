"""One-command entry point for the complete Section 6.4 benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def print_banner(message: str):
    line = "=" * 88
    print(f"\n{line}", flush=True)
    print(message, flush=True)
    print(line, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=["a", "b"], default=["a", "b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner(
        "STARTING COMPLETE BENCHMARK | "
        f"scenarios={','.join(value.upper() for value in args.scenarios)} | "
        f"replications={len(args.seeds)}"
    )
    run_command = [
        sys.executable,
        "run_benchmark_experiments.py",
        "--scenarios",
        *args.scenarios,
        "--seeds",
        *(str(seed) for seed in args.seeds),
    ]
    subprocess.run(run_command, cwd=SCRIPT_DIR, check=True)

    for scenario in args.scenarios:
        print_banner(f"STARTING AGGREGATION AND FIGURES | scenario={scenario.upper()}")
        subprocess.run(
            [sys.executable, "summarize_benchmark_replications.py", "--scenario", scenario],
            cwd=SCRIPT_DIR,
            check=True,
        )
        print_banner(f"COMPLETED AGGREGATION AND FIGURES | scenario={scenario.upper()}")

    print_banner("COMPLETE BENCHMARK FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
