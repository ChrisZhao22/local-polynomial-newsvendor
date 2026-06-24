"""Aggregate fixed-split benchmark replications and create publication figures."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, t


RUNS_DIR = ROOT_DIR / "Data" / "BenchmarkComparison"
SUMMARY_DIR = RUNS_DIR / "summary"
FIGURE_DIR = ROOT_DIR / "Figure" / "BenchmarkComparison"
MODELS = {
    "Est-Opt (OLS)": "nv_ETO",
    "KO": "nv_KO",
    "LinearModel": "nv_LinearModel",
    "LinearModel-L1": "nv_LinearModel_L1",
    "LinearModel-L2": "nv_LinearModel_L2",
    "SAA": "nv_SAA",
    "DNN": "nv_DNN",
    "Minimax (Scarf)": "nv_scarf",
    "RKHS": "nv_RKHS",
    "KPQR": "nv_KPQR",
    "Oracle": "nv_Oracle",
}
RUNTIME_NAMES = {
    "DNN": "DNN (Deep_NV)",
    "Est-Opt (OLS)": "Est-Opt (OLS)",
    "KO": "KO",
    "KPQR": "KPQR",
    "LinearModel": "LinearModel",
    "LinearModel-L1": "LinearModel-L1",
    "LinearModel-L2": "LinearModel-L2",
    "Minimax (Scarf)": "Minimax (Scarf)",
    "Oracle": "Oracle",
    "RKHS": "RKHS",
    "SAA": "SAA",
}
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#4C72B0",
]
LINE_STYLES = ["-", "--", "-.", ":"] * 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["a", "b"], required=True)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional stable figure tag; the default is the current timestamp.",
    )
    return parser.parse_args()


def ci_half_width(values):
    values = np.asarray(values, dtype=float)
    return 0.0 if len(values) <= 1 else float(sem(values) * t.ppf(0.975, len(values) - 1))


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.titlesize": 16,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 10.2,
        "axes.unicode_minus": False,
    })


def load_replications(scenario):
    records = []
    trajectories = {}
    runtimes = []
    scenario_root = RUNS_DIR / f"scenario_{scenario}"
    for seed_dir in sorted(scenario_root.glob("seed_*")):
        if not (seed_dir / ".complete").exists():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        output_dir = seed_dir / f"scenario_{scenario}"
        seed_costs = {}
        for model, stem in MODELS.items():
            path = output_dir / f"{stem}_scenario_{scenario}.csv"
            if not path.exists():
                continue
            costs = pd.read_csv(path)["Operation_Cost"].to_numpy(dtype=float)
            seed_costs[model] = costs
            trajectories.setdefault(model, []).append(np.cumsum(costs))
            records.append({"scenario": scenario, "seed": seed, "model_name": model, "mean_test_cost": np.mean(costs)})
        runtime_path = output_dir / f"runtime_metrics_scenario_{scenario}.csv"
        if runtime_path.exists():
            runtime = pd.read_csv(runtime_path)
            for model, runtime_name in RUNTIME_NAMES.items():
                match = runtime[runtime["model_name"] == runtime_name]
                if not match.empty:
                    row = match.iloc[-1]
                    runtimes.append({
                        "scenario": scenario,
                        "seed": seed,
                        "model_name": model,
                        "total_test_runtime_sec": row["total_test_runtime_sec"],
                        "per_sample_runtime_sec": row["per_sample_runtime_sec"],
                    })
    return pd.DataFrame(records), trajectories, pd.DataFrame(runtimes)


def build_summaries(raw):
    rows = []
    pivot = raw.pivot(index="seed", columns="model_name", values="mean_test_cost")
    if "Oracle" not in pivot or "KPQR" not in pivot:
        raise ValueError("Complete Oracle and KPQR replications are required.")
    for model in pivot.columns:
        values = pivot[model].dropna()
        common_oracle = pivot.loc[values.index, "Oracle"]
        oracle_gap = values - common_oracle
        common_kpqr = pivot.loc[values.index, "KPQR"]
        versus_kpqr = values - common_kpqr
        rows.append({
            "model_name": model,
            "replications": len(values),
            "mean_test_cost": values.mean(),
            "ci95_test_cost": ci_half_width(values),
            "mean_excess_over_oracle": oracle_gap.mean(),
            "ci95_excess_over_oracle": ci_half_width(oracle_gap),
            "mean_paired_difference_vs_kpqr": versus_kpqr.mean(),
            "ci95_paired_difference_vs_kpqr": ci_half_width(versus_kpqr),
            "kpqr_win_rate": float(np.mean(versus_kpqr > 0.0)) if model != "KPQR" else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_test_cost").reset_index(drop=True)


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost(summary, trajectories, scenario, timestamp):
    apply_style()
    ordered = summary.sort_values("mean_test_cost", ascending=False)
    models = ordered["model_name"].tolist()
    colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(models)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 6.1))
    y = np.arange(len(models))
    ax1.barh(
        y,
        ordered["mean_test_cost"],
        xerr=ordered["ci95_test_cost"],
        color=[colors[m] for m in models],
        alpha=0.82,
        height=0.62,
        error_kw={"ecolor": "#333333", "capsize": 3.5, "elinewidth": 1.05},
    )
    ax1.set_yticks(y, models)
    ax1.set_xlabel("Mean test cost (95% CI across replications)")
    ax1.set_title("Out-of-Sample Operating Cost", pad=10)
    ax1.set_xlim(left=max(0.0, float(np.min(ordered["mean_test_cost"] - ordered["ci95_test_cost"])) * 0.95))

    for i, model in enumerate(summary.sort_values("mean_test_cost")["model_name"]):
        curves = np.asarray(trajectories.get(model, []), dtype=float)
        if curves.size == 0:
            continue
        mean_curve = curves.mean(axis=0)
        x = np.arange(1, len(mean_curve) + 1)
        half = np.asarray([ci_half_width(curves[:, j]) for j in range(curves.shape[1])])
        ax2.plot(
            x,
            mean_curve,
            label=model,
            color=colors[model],
            linestyle=LINE_STYLES[i],
            linewidth=1.9,
        )
        ax2.fill_between(x, mean_curve - half, mean_curve + half, color=colors[model], alpha=0.07)
    ax2.set_xlabel("Test observation")
    ax2.set_ylabel("Mean cumulative cost")
    ax2.set_title("Cumulative Cost Trajectories", pad=10)
    ax2.legend(loc="upper left", ncol=2, fontsize=9.6, columnspacing=1.1, handlelength=2.2)
    fig.tight_layout(w_pad=2.1)
    figure_tag = timestamp
    base = FIGURE_DIR / f"Benchmark_Cost_Comparison_scenario_{scenario}_{figure_tag}"
    save_figure(fig, base)


def plot_cost_bar_only(summary, scenario, timestamp):
    apply_style()
    ordered = summary.sort_values("mean_test_cost", ascending=False)
    models = ordered["model_name"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.barh(
        y,
        ordered["mean_test_cost"],
        xerr=ordered["ci95_test_cost"],
        color=colors,
        alpha=0.82,
        height=0.62,
        error_kw={"ecolor": "#333333", "capsize": 3},
    )
    ax.set_yticks(y, models)
    ax.set_xlabel("Mean test newsvendor cost (95% CI across replications)")
    ax.set_title("Out-of-Sample Newsvendor Cost")
    ax.set_xlim(left=max(0.0, float(np.min(ordered["mean_test_cost"] - ordered["ci95_test_cost"])) * 0.95))
    ax.text(
        0.98,
        0.96,
        f"Scenario {scenario.upper()}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"Benchmark_Cost_Bars_scenario_{scenario}_{timestamp}")


def main():
    args = parse_args()
    timestamp = args.output_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw, trajectories, runtimes = load_replications(args.scenario)
    if raw.empty:
        raise ValueError(f"No complete replications found for scenario {args.scenario}.")
    summary = build_summaries(raw)
    raw.to_csv(SUMMARY_DIR / f"benchmark_replication_costs_scenario_{args.scenario}.csv", index=False)
    summary.to_csv(SUMMARY_DIR / f"benchmark_summary_scenario_{args.scenario}.csv", index=False)
    if not runtimes.empty:
        runtimes.to_csv(SUMMARY_DIR / f"benchmark_runtime_replications_scenario_{args.scenario}.csv", index=False)
    plot_cost(summary, trajectories, args.scenario, timestamp)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
