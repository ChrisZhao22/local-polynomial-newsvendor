"""Polynomial hierarchy experiment for Section 6.3.

Degree zero is KO, while degrees one and two are higher-order KPQR policies.
The default experiment holds the bandwidth fixed across degrees so that it
isolates the approximation-order effect in Theorem 1.  Supplying multiple
bandwidths enables a secondary holdout-tuned robustness experiment.  The final
comparison uses exact conditional newsvendor risk on a common test grid,
followed by paired comparisons over independent simulation replications.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
RATE_VALIDATION_DIR = SCRIPT_DIR.parent / "RateValidation"
if str(RATE_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(RATE_VALIDATION_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from rate_validation import exact_excess_risk, fit_kpqr_at_query


CONFIG_PATH = SCRIPT_DIR / "polynomial_hierarchy_config.json"
DATA_DIR = (ROOT_DIR / "Data" / "PolynomialHierarchy").resolve()
FIGURE_DIR = (ROOT_DIR / "Figure" / "PolynomialHierarchy").resolve()


def parse_args() -> argparse.Namespace:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Compare degree-0, degree-1, and degree-2 KPQR policies."
    )
    parser.add_argument("--training-size", type=int, default=config["training_size"])
    parser.add_argument("--validation-size", type=int, default=config["validation_size"])
    parser.add_argument("--test-grid-size", type=int, default=config["test_grid_size"])
    parser.add_argument("--replications", type=int, default=config["replications"])
    parser.add_argument("--degrees", type=int, nargs="+", default=config["degrees"])
    parser.add_argument("--bandwidths", type=float, nargs="+", default=config["bandwidths"])
    parser.add_argument("--beta", type=float, default=config["beta"])
    parser.add_argument(
        "--fractional-coefficient", type=float, default=config["fractional_coefficient"]
    )
    parser.add_argument("--fractional-knot", type=float, default=config["fractional_knot"])
    parser.add_argument("--tau", type=float, default=config["tau"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument(
        "--kernel",
        choices=["epanechnikov", "gaussian"],
        default=config.get("kernel", "epanechnikov"),
    )
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    if args.training_size <= 0 or args.validation_size <= 0:
        parser.error("Training and validation sizes must be positive.")
    if args.test_grid_size < 3:
        parser.error("--test-grid-size must be at least 3.")
    if args.replications < 2:
        parser.error("--replications must be at least 2.")
    if sorted(set(args.degrees)) != [0, 1, 2]:
        parser.error("This hierarchy experiment requires exactly degrees 0, 1, and 2.")
    if not 2.0 < args.beta < 3.0:
        parser.error("This experiment requires beta in (2, 3), so floor(beta)=2.")
    if args.fractional_coefficient <= 0:
        parser.error("--fractional-coefficient must be positive.")
    if not 0 < args.fractional_knot < 1:
        parser.error("--fractional-knot must be an interior point of [0, 1].")
    if not args.bandwidths or any(value <= 0 for value in args.bandwidths):
        parser.error("All candidate bandwidths must be positive.")
    if not 0 < args.tau < 1:
        parser.error("--tau must lie strictly between 0 and 1.")

    args.degrees = [0, 1, 2]
    args.bandwidths = sorted(set(args.bandwidths))
    return args


def oracle_quantile(
    x: np.ndarray,
    beta: float = 2.5,
    fractional_coefficient: float = 2.0,
    fractional_knot: float = 0.5,
) -> np.ndarray:
    """A genuinely beta-Hölder prescription with floor(beta)=2."""
    x = np.asarray(x, dtype=float)
    return (
        2.0
        + 1.5 * x
        + 6.0 * x**2
        + fractional_coefficient * np.abs(x - fractional_knot) ** beta
    )


def generate_sample(
    size: int,
    tau: float,
    beta: float,
    fractional_coefficient: float,
    fractional_knot: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, size=size)
    noise = rng.uniform(0.0, 1.0, size=size) - tau
    demand = oracle_quantile(x, beta, fractional_coefficient, fractional_knot) + noise
    return x, demand


def newsvendor_cost(decision: np.ndarray, demand: np.ndarray, tau: float) -> np.ndarray:
    decision = np.asarray(decision, dtype=float)
    demand = np.asarray(demand, dtype=float)
    return tau * np.maximum(demand - decision, 0.0) + (1.0 - tau) * np.maximum(
        decision - demand, 0.0
    )


def predict_policy(
    x_train: np.ndarray,
    demand_train: np.ndarray,
    query_points: np.ndarray,
    bandwidth: float,
    degree: int,
    tau: float,
    kernel: str,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(len(query_points), dtype=float)
    effective_sizes = np.empty(len(query_points), dtype=float)
    for index, query in enumerate(query_points):
        prediction, effective_n = fit_kpqr_at_query(
            x_train=x_train,
            demand=demand_train,
            query_point=float(query),
            bandwidth=bandwidth,
            degree=degree,
            tau=tau,
            kernel=kernel,
        )
        predictions[index] = prediction
        effective_sizes[index] = effective_n
    return predictions, effective_sizes


def exact_grid_excess_risk(predictions: np.ndarray, oracle: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(
        [exact_excess_risk(float(q_hat), float(q_star), tau) for q_hat, q_star in zip(predictions, oracle)],
        dtype=float,
    )


def run_experiment(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_records: list[dict] = []
    selection_records: list[dict] = []
    representative_records: list[dict] = []
    test_grid = np.linspace(0.02, 0.98, args.test_grid_size)
    oracle_grid = oracle_quantile(
        test_grid,
        args.beta,
        args.fractional_coefficient,
        args.fractional_knot,
    )
    oracle_risk = args.tau * (1.0 - args.tau) / 2.0

    for replication in range(args.replications):
        print(f"Replication {replication + 1}/{args.replications}")
        train_rng = np.random.default_rng(np.random.SeedSequence([args.seed, replication, 0]))
        validation_rng = np.random.default_rng(np.random.SeedSequence([args.seed, replication, 1]))
        x_train, demand_train = generate_sample(
            args.training_size,
            args.tau,
            args.beta,
            args.fractional_coefficient,
            args.fractional_knot,
            train_rng,
        )
        x_validation, demand_validation = generate_sample(
            args.validation_size,
            args.tau,
            args.beta,
            args.fractional_coefficient,
            args.fractional_knot,
            validation_rng,
        )

        for degree in args.degrees:
            degree_start = time.time()
            degree_selection: list[dict] = []
            for bandwidth in args.bandwidths:
                validation_predictions, validation_effective_n = predict_policy(
                    x_train=x_train,
                    demand_train=demand_train,
                    query_points=x_validation,
                    bandwidth=bandwidth,
                    degree=degree,
                    tau=args.tau,
                    kernel=args.kernel,
                )
                validation_loss = float(
                    np.mean(newsvendor_cost(validation_predictions, demand_validation, args.tau))
                )
                record = {
                    "replication": replication,
                    "degree": degree,
                    "method": "KO" if degree == 0 else f"KPQR-{degree}",
                    "bandwidth": bandwidth,
                    "validation_mean_nv_loss": validation_loss,
                    "validation_mean_effective_n": float(np.mean(validation_effective_n)),
                }
                selection_records.append(record)
                degree_selection.append(record)

            best = min(degree_selection, key=lambda row: row["validation_mean_nv_loss"])
            selected_bandwidth = float(best["bandwidth"])
            test_predictions, test_effective_n = predict_policy(
                x_train=x_train,
                demand_train=demand_train,
                query_points=test_grid,
                bandwidth=selected_bandwidth,
                degree=degree,
                tau=args.tau,
                kernel=args.kernel,
            )
            pointwise_excess = exact_grid_excess_risk(test_predictions, oracle_grid, args.tau)
            mean_excess = float(np.mean(pointwise_excess))
            result_records.append(
                {
                    "replication": replication,
                    "degree": degree,
                    "method": "KO" if degree == 0 else f"KPQR-{degree}",
                    "selected_bandwidth": selected_bandwidth,
                    "validation_mean_nv_loss": float(best["validation_mean_nv_loss"]),
                    "test_mean_excess_risk": mean_excess,
                    "test_expected_nv_loss": oracle_risk + mean_excess,
                    "test_mean_effective_n": float(np.mean(test_effective_n)),
                    "runtime_sec": time.time() - degree_start,
                }
            )

            if replication == 0:
                for query, oracle_value, prediction, excess in zip(
                    test_grid, oracle_grid, test_predictions, pointwise_excess
                ):
                    representative_records.append(
                        {
                            "replication": replication,
                            "degree": degree,
                            "method": "KO" if degree == 0 else f"KPQR-{degree}",
                            "selected_bandwidth": selected_bandwidth,
                            "query_x": query,
                            "oracle_decision": oracle_value,
                            "estimated_decision": prediction,
                            "pointwise_excess_risk": excess,
                        }
                    )

    return (
        pd.DataFrame(result_records),
        pd.DataFrame(selection_records),
        pd.DataFrame(representative_records),
    )


def mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(len(values))
    critical = float(student_t.ppf(0.975, len(values) - 1))
    return mean, std, mean - critical * se, mean + critical * se


def summarize_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for degree, group in result_df.groupby("degree", sort=True):
        mean, std, lower, upper = mean_ci(group["test_mean_excess_risk"].to_numpy())
        expected_mean, expected_std, expected_lower, expected_upper = mean_ci(
            group["test_expected_nv_loss"].to_numpy()
        )
        rows.append(
            {
                "degree": int(degree),
                "method": group["method"].iloc[0],
                "replications": len(group),
                "mean_selected_bandwidth": float(group["selected_bandwidth"].mean()),
                "median_selected_bandwidth": float(group["selected_bandwidth"].median()),
                "mean_test_excess_risk": mean,
                "std_test_excess_risk": std,
                "ci95_excess_lower": max(lower, 0.0),
                "ci95_excess_upper": upper,
                "mean_expected_nv_loss": expected_mean,
                "std_expected_nv_loss": expected_std,
                "ci95_nv_loss_lower": expected_lower,
                "ci95_nv_loss_upper": expected_upper,
                "mean_runtime_sec": float(group["runtime_sec"].mean()),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("degree").reset_index(drop=True)
    summary_df["improvement_vs_previous_pct"] = np.nan
    for index in range(1, len(summary_df)):
        previous = summary_df.loc[index - 1, "mean_test_excess_risk"]
        current = summary_df.loc[index, "mean_test_excess_risk"]
        summary_df.loc[index, "improvement_vs_previous_pct"] = 100.0 * (previous - current) / previous

    paired_rows: list[dict] = []
    pivot = result_df.pivot(index="replication", columns="degree", values="test_mean_excess_risk")
    for lower_degree, higher_degree in [(0, 1), (1, 2), (0, 2)]:
        differences = pivot[lower_degree].to_numpy() - pivot[higher_degree].to_numpy()
        mean, std, ci_lower, ci_upper = mean_ci(differences)
        paired_rows.append(
            {
                "comparison": f"degree_{lower_degree}_minus_degree_{higher_degree}",
                "mean_excess_risk_reduction": mean,
                "std_reduction": std,
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
                "fraction_higher_degree_better": float(np.mean(differences > 0.0)),
            }
        )
    return summary_df, pd.DataFrame(paired_rows)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hierarchy(summary_df: pd.DataFrame, output_base: Path) -> None:
    apply_plot_style()
    x = np.arange(len(summary_df))
    means = summary_df["mean_test_excess_risk"].to_numpy(dtype=float)
    lower = summary_df["ci95_excess_lower"].to_numpy(dtype=float)
    upper = summary_df["ci95_excess_upper"].to_numpy(dtype=float)
    errors = np.vstack([means - lower, upper - means])
    labels = ["Degree 0\n(KO)", "Degree 1\n(KPQR)", "Degree 2\n(KPQR)"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7.5, 5.25))
    ax.plot(x, means, color="#555555", linewidth=1.45, zorder=2)
    for index, color in enumerate(colors):
        ax.errorbar(
            x[index],
            means[index],
            yerr=errors[:, index].reshape(2, 1),
            fmt="none",
            ecolor=color,
            elinewidth=2.2,
            capsize=8,
            capthick=2.0,
            zorder=4,
        )
        ax.plot(
            x[index],
            means[index],
            marker="o",
            linestyle="none",
            markersize=6.2,
            color=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=5,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean conditional excess risk")
    ax.set_yscale("log")
    ax.margins(x=0.18, y=0.22)
    ax.grid(True, which="major", axis="y")
    ax.grid(False, axis="x")
    annotation_specs = [
        (0, 20, "center"),
        (0, 24, "center"),
        (18, 17, "left"),
    ]
    for index, mean in enumerate(means):
        dx, dy, ha = annotation_specs[index]
        ax.annotate(
            f"{mean:.2e}",
            xy=(x[index], mean),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=14,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.8},
            zorder=6,
        )
    fig.tight_layout(pad=0.9)
    save_figure(fig, output_base)


def plot_representative_policies(representative_df: pd.DataFrame, output_base: Path) -> None:
    apply_plot_style()
    colors = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52"}
    line_styles = {0: "--", 1: "-.", 2: ":"}
    fig, ax = plt.subplots(figsize=(7.5, 5.25))
    oracle = (
        representative_df[["query_x", "oracle_decision"]]
        .drop_duplicates(subset=["query_x"])
        .sort_values("query_x")
    )
    ax.plot(
        oracle["query_x"],
        oracle["oracle_decision"],
        color="#222222",
        linewidth=2.4,
        label=r"Oracle $q^*(x)$",
    )
    for degree, group in representative_df.groupby("degree", sort=True):
        group = group.sort_values("query_x")
        label = "KO (degree 0)" if degree == 0 else f"KPQR degree {degree}"
        ax.plot(
            group["query_x"],
            group["estimated_decision"],
            color=colors[int(degree)],
            linewidth=2.1,
            linestyle=line_styles[int(degree)],
            label=label,
        )
    ax.set_xlabel(r"Context $x$")
    ax.set_ylabel("Newsvendor decision")
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout(pad=0.9)
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    timestamp = args.output_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("Polynomial hierarchy beyond KO")
    print(
        "Oracle quantile: q*(x) = 2 + 1.5x + 6x^2 + "
        f"{args.fractional_coefficient:g}|x-{args.fractional_knot:g}|^{args.beta:g}"
    )
    print(
        f"Hölder smoothness beta={args.beta}; "
        f"highest KPQR degree=floor(beta)={max(args.degrees)}"
    )
    print(
        f"Training/validation/test-grid: {args.training_size}/"
        f"{args.validation_size}/{args.test_grid_size}"
    )
    print(f"Replications: {args.replications}; tau={args.tau}; kernel={args.kernel}")
    print(f"Bandwidth candidates: {args.bandwidths}")

    result_df, selection_df, representative_df = run_experiment(args)
    summary_df, paired_df = summarize_results(result_df)

    result_path = DATA_DIR / f"polynomial_hierarchy_replications_{timestamp}.csv"
    selection_path = DATA_DIR / f"polynomial_hierarchy_bandwidth_selection_{timestamp}.csv"
    representative_path = DATA_DIR / f"polynomial_hierarchy_representative_policy_{timestamp}.csv"
    summary_path = DATA_DIR / f"polynomial_hierarchy_summary_{timestamp}.csv"
    paired_path = DATA_DIR / f"polynomial_hierarchy_paired_comparisons_{timestamp}.csv"
    metadata_path = DATA_DIR / f"polynomial_hierarchy_metadata_{timestamp}.json"
    figure_tag = timestamp
    hierarchy_figure = FIGURE_DIR / f"Polynomial_Hierarchy_{figure_tag}"
    policy_figure = FIGURE_DIR / f"Polynomial_Hierarchy_Policies_{figure_tag}"

    result_df.to_csv(result_path, index=False)
    selection_df.to_csv(selection_path, index=False)
    representative_df.to_csv(representative_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    metadata = {
        "experiment": "polynomial_hierarchy_beyond_ko",
        "oracle_quantile": (
            "2 + 1.5*x + 6*x^2 + "
            f"{args.fractional_coefficient}*abs(x-{args.fractional_knot})^{args.beta}"
        ),
        "beta": args.beta,
        "fractional_coefficient": args.fractional_coefficient,
        "fractional_knot": args.fractional_knot,
        "noise": "Uniform(0,1)-tau",
        "training_size": args.training_size,
        "validation_size": args.validation_size,
        "test_grid_size": args.test_grid_size,
        "replications": args.replications,
        "degrees": args.degrees,
        "bandwidths": args.bandwidths,
        "tau": args.tau,
        "kernel": args.kernel,
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    plot_hierarchy(summary_df, hierarchy_figure)
    plot_representative_policies(representative_df, policy_figure)

    print(summary_df.to_string(index=False))
    print("\nPaired comparisons:")
    print(paired_df.to_string(index=False))
    print(f"Summary: {summary_path}")
    print(f"Paired comparisons: {paired_path}")
    print(f"Hierarchy figure: {hierarchy_figure.with_suffix('.pdf')}")
    print(f"Policy figure: {policy_figure.with_suffix('.pdf')}")
    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
