"""Compare simulation-study model outputs using the CaseStudy comparison style."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as st

from simulation_paths import scenario_csv_path


DATA_DIR = (ROOT_DIR / "Data" / "BenchmarkComparison").resolve()
FIGURE_DIR = (ROOT_DIR / "Figure" / "BenchmarkComparison").resolve()
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"
MODEL_FILENAMES = {
    "Est-Opt (OLS)": "nv_ETO.csv",
    "KO": "nv_KO.csv",
    "LinearModel": "nv_LinearModel.csv",
    "LinearModel-L1": "nv_LinearModel_L1.csv",
    "LinearModel-L2": "nv_LinearModel_L2.csv",
    "SAA": "nv_SAA.csv",
    "DNN": "nv_DNN.csv",
    "Minimax (Scarf)": "nv_scarf.csv",
    "RKHS": "nv_RKHS.csv",
    "KPQR": "nv_KPQR.csv",
    "Oracle": "nv_Oracle.csv",
}

# Same visual language as CaseStudy/Algorithm/comparison1.py.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
LINE_STYLES = ["-", "--", "-.", ":"] * 3


def _apply_academic_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial Unicode MS",
            "PingFang SC",
            "Hiragino Sans GB",
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def _sanitize_scenario(scenario: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in scenario)
    return clean or "unknown"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _confidence_interval_95(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(st.sem(values) * st.t.ppf((1 + 0.95) / 2.0, len(values) - 1))


def _load_results(max_len: int | None, config: dict) -> dict[str, np.ndarray]:
    print("Reading model outputs...")
    results: dict[str, np.ndarray] = {}

    for model_name, filename in MODEL_FILENAMES.items():
        file_path = scenario_csv_path(filename, config)
        if not file_path.exists():
            print(f"Warning: file not found, skipped {model_name}: {file_path}")
            continue

        df_model = pd.read_csv(file_path)
        if "Operation_Cost" not in df_model.columns:
            print(f"Warning: missing Operation_Cost, skipped {model_name}: {file_path}")
            continue

        costs = pd.to_numeric(df_model["Operation_Cost"], errors="coerce").dropna().to_numpy(dtype=float)
        if max_len is not None and len(costs) > max_len:
            costs = costs[:max_len]

        results[model_name] = costs
        print(f"- Loaded {model_name}: {len(costs)} rows")

    if not results:
        raise ValueError("No model output files were loaded.")

    return results


def _build_summary(results: dict[str, np.ndarray]) -> pd.DataFrame:
    summary_data = []
    for name, costs in results.items():
        summary_data.append({
            "model_name": name,
            "Total Cost": float(np.sum(costs)),
            "Mean Cost": float(np.mean(costs)),
            "Std Dev": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
            "CI": _confidence_interval_95(costs),
            "Min Cost": float(np.min(costs)),
            "Max Cost": float(np.max(costs)),
            "N": int(len(costs)),
        })

    # Horizontal bars render bottom-to-top, so descending puts the best model at the top.
    return pd.DataFrame(summary_data).sort_values("Mean Cost", ascending=False).reset_index(drop=True)


def _save_summary(summary_df: pd.DataFrame, config: dict, timestamp: str) -> None:
    scenario_value = str(config.get("scenario", "unknown"))

    snapshot_file = scenario_csv_path(f"comparison_performance_{timestamp}.csv", config)
    history_file = scenario_csv_path("comparison_performance_history.csv", config)

    save_df = summary_df.sort_values("Mean Cost", ascending=True).reset_index(drop=True)
    save_df.insert(0, "rank_by_mean_cost", np.arange(1, len(save_df) + 1))
    save_df.insert(0, "tau", config.get("tau", np.nan))
    save_df.insert(0, "scenario", scenario_value)
    save_df.insert(0, "timestamp", timestamp)

    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(snapshot_file, index=False)
    write_header = not history_file.exists() or history_file.stat().st_size == 0
    save_df.to_csv(history_file, mode="a", header=write_header, index=False)

    print(f"Performance snapshot saved: {snapshot_file}")
    print(f"Performance history updated: {history_file}")


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_comparison(
    results: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    scenario_tag: str,
    timestamp: str,
) -> Path:
    ordered_models = summary_df["model_name"].tolist()
    color_map = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(ordered_models)}
    ls_map = {model: LINE_STYLES[i % len(LINE_STYLES)] for i, model in enumerate(ordered_models)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.3)

    mean_costs = summary_df["Mean Cost"].to_numpy()
    cis = summary_df["CI"].to_numpy()
    y_pos = np.arange(len(ordered_models))
    bar_colors = [color_map[m] for m in ordered_models]

    ax1.barh(
        y_pos,
        mean_costs,
        xerr=cis,
        align="center",
        color=bar_colors,
        alpha=0.8,
        edgecolor="none",
        height=0.6,
        error_kw=dict(ecolor="#333333", lw=1.2, capsize=3, capthick=1.2),
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ordered_models)
    ax1.set_xlabel("Mean test cost (95% confidence interval)")
    ax1.set_title("Out-of-Sample Operating Cost")
    ax1.set_xlim(left=max(0, np.min(mean_costs - cis) * 0.95))

    best_to_worst_models = summary_df.sort_values("Mean Cost", ascending=True)["model_name"].tolist()
    for model_name in best_to_worst_models:
        costs = results[model_name]
        step = max(1, len(costs) // 200)
        x_val = np.arange(0, len(costs), step)
        y_val = np.cumsum(costs)[::step]
        ax2.plot(
            x_val,
            y_val,
            label=model_name,
            color=color_map[model_name],
            linestyle=ls_map[model_name],
            linewidth=1.8,
            alpha=0.85,
        )

    ax2.set_xlabel("Test observation")
    ax2.set_ylabel("Cumulative cost")
    ax2.set_title("Cumulative Cost Trajectories")
    ax2.legend(loc="upper left", ncol=2, frameon=False, fontsize=9.5)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    base_path = FIGURE_DIR / f"Cost_Comparison_{scenario_tag}_{timestamp}"
    _save_figure(fig, base_path)
    return base_path.with_suffix(".pdf")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    scenario_value = str(config.get("scenario", "unknown"))
    scenario_tag = f"scenario_{_sanitize_scenario(scenario_value)}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    max_len = int(config["lnte"]) if config.get("lnte") is not None else None

    results = _load_results(max_len=max_len, config=config)
    summary_df = _build_summary(results)

    print("\n" + "=" * 72)
    print("Model Ranking by Mean Cost (Lower is Better)")
    print("=" * 72)
    print(summary_df.sort_values("Mean Cost", ascending=True).reset_index(drop=True))
    print("=" * 72)

    _save_summary(summary_df, config, timestamp)
    _apply_academic_style()

    cost_figure_path = _plot_cost_comparison(results, summary_df, scenario_tag, timestamp)
    print(f"Cost comparison figure saved: {cost_figure_path}")


if __name__ == "__main__":
    main()
