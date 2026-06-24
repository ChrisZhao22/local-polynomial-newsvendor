"""Log-log validation of the KPQR contextwise excess-risk rate.

The experiment targets Theorem 2 using a one-dimensional beta-Hölder
conditional quantile.  At every sample size, independent training samples are
used to estimate the conditional tau-quantile at one or more interior query
points.
Because the noise is uniform, conditional newsvendor risk is evaluated exactly
rather than through a noisy finite test set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
from scipy.stats import t as student_t
from sklearn.linear_model import QuantileRegressor


CONFIG_PATH = SCRIPT_DIR / "rate_validation_config.json"
DATA_DIR = (ROOT_DIR / "Data" / "RateValidation").resolve()
FIGURE_DIR = (ROOT_DIR / "Figure" / "RateValidation").resolve()


def parse_args() -> argparse.Namespace:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Validate the KPQR excess-risk rate on log-log axes."
    )
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=config["sample_sizes"])
    parser.add_argument("--replications", type=int, default=config["replications"])
    parser.add_argument("--beta", type=float, default=config["beta"])
    parser.add_argument("--degree", type=int, default=config["polynomial_degree"])
    parser.add_argument("--tau", type=float, default=config["tau"])
    parser.add_argument("--query-point", type=float, default=config["query_point"])
    parser.add_argument(
        "--query-points",
        type=float,
        nargs="+",
        default=config.get("query_points", [config["query_point"]]),
        help="Interior query points for pointwise rate validation.",
    )
    parser.add_argument(
        "--dgp",
        choices=["centered_query", "knot_sum"],
        default=config.get("dgp", "centered_query"),
        help=(
            "centered_query uses the original query-centered Hölder function; "
            "knot_sum uses a fixed DGP with Hölder knots."
        ),
    )
    parser.add_argument(
        "--holder-knots",
        type=float,
        nargs="+",
        default=config.get("holder_knots", config.get("query_points", [config["query_point"]])),
        help="Fixed knots used when --dgp=knot_sum.",
    )
    parser.add_argument(
        "--holder-coefficient",
        type=float,
        default=config.get("holder_coefficient", 2.0),
        help="Coefficient multiplying the Hölder knot terms.",
    )
    parser.add_argument(
        "--bandwidth-constant", type=float, default=config["bandwidth_constant"]
    )
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument(
        "--kernel",
        choices=["epanechnikov", "gaussian"],
        default=config.get("kernel", "epanechnikov"),
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional stable output tag; the default is a timestamp.",
    )
    args = parser.parse_args()

    if not args.sample_sizes or any(n <= 0 for n in args.sample_sizes):
        parser.error("--sample-sizes must contain positive integers.")
    if args.replications < 2:
        parser.error("--replications must be at least 2.")
    if args.beta <= 0:
        parser.error("--beta must be positive.")
    if args.degree < 0:
        parser.error("--degree must be non-negative.")
    if args.degree != int(np.floor(args.beta)):
        parser.error("The theorem-aligned degree must equal floor(beta).")
    if not 0 < args.tau < 1:
        parser.error("--tau must lie strictly between 0 and 1.")
    if not 0 < args.query_point < 1:
        parser.error("--query-point must be an interior point of [0, 1].")
    if not args.query_points or any(not 0 < point < 1 for point in args.query_points):
        parser.error("--query-points must contain interior points of [0, 1].")
    if not args.holder_knots or any(not 0 < knot < 1 for knot in args.holder_knots):
        parser.error("--holder-knots must contain interior points of [0, 1].")
    if args.holder_coefficient <= 0:
        parser.error("--holder-coefficient must be positive.")
    if args.bandwidth_constant <= 0:
        parser.error("--bandwidth-constant must be positive.")

    args.sample_sizes = sorted(set(args.sample_sizes))
    args.query_points = sorted(set(args.query_points))
    args.holder_knots = sorted(set(args.holder_knots))
    return args


def oracle_quantile(
    x: np.ndarray,
    query_point: float,
    beta: float,
    dgp: str = "centered_query",
    holder_knots: list[float] | None = None,
    holder_coefficient: float = 2.0,
) -> np.ndarray:
    """A beta-Hölder quantile with a genuine non-polynomial beta-order term."""
    x_arr = np.asarray(x, dtype=float)
    if dgp == "centered_query":
        z = x_arr - query_point
        return 2.0 + 0.5 * z + 0.75 * z**2 + 2.0 * np.abs(z) ** beta

    holder_knots = holder_knots or [0.25, 0.5, 0.75]
    holder_term = np.zeros_like(x_arr, dtype=float)
    for knot in holder_knots:
        holder_term += np.abs(x_arr - knot) ** beta
    return 2.0 + 0.5 * x_arr + 0.75 * x_arr**2 + holder_coefficient * holder_term


def kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    abs_u = np.abs(u)
    if kernel == "epanechnikov":
        return 0.75 * np.maximum(1.0 - u**2, 0.0)
    return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, tau: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = tau * np.sum(sorted_weights)
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def fit_kpqr_at_query(
    x_train: np.ndarray,
    demand: np.ndarray,
    query_point: float,
    bandwidth: float,
    degree: int,
    tau: float,
    kernel: str,
) -> tuple[float, int]:
    centered = (x_train - query_point) / bandwidth
    weights = kernel_weights(centered, kernel)
    mask = weights > 1e-10
    effective_n = int(np.sum(mask))
    if effective_n <= degree + 1:
        return weighted_quantile(demand, np.ones_like(demand), tau), effective_n

    centered = centered[mask]
    demand_eff = demand[mask]
    weights_eff = weights[mask]
    design = np.column_stack([centered**power for power in range(degree + 1)])

    model = QuantileRegressor(
        quantile=tau,
        alpha=0.0,
        fit_intercept=False,
        solver="highs",
    )
    try:
        model.fit(design, demand_eff, sample_weight=weights_eff)
        estimate = float(model.coef_[0])
    except (ValueError, RuntimeError):
        estimate = weighted_quantile(demand_eff, weights_eff, tau)
    return estimate, effective_n


def positive_part_expectation(delta: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return E[(epsilon-delta)+] for epsilon ~ Uniform(lower, upper)."""
    delta = np.asarray(delta, dtype=float)
    mean_epsilon = 0.5 * (lower + upper)
    result = np.zeros_like(delta)
    below = delta <= lower
    middle = (delta > lower) & (delta < upper)
    result[below] = mean_epsilon - delta[below]
    result[middle] = (upper - delta[middle]) ** 2 / (2.0 * (upper - lower))
    return result


def negative_part_expectation(delta: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return E[(delta-epsilon)+] for epsilon ~ Uniform(lower, upper)."""
    delta = np.asarray(delta, dtype=float)
    mean_epsilon = 0.5 * (lower + upper)
    result = np.zeros_like(delta)
    middle = (delta > lower) & (delta < upper)
    above = delta >= upper
    result[middle] = (delta[middle] - lower) ** 2 / (2.0 * (upper - lower))
    result[above] = delta[above] - mean_epsilon
    return result


def exact_excess_risk(estimate: float, oracle: float, tau: float) -> float:
    """Exact conditional newsvendor excess risk under U[-tau, 1-tau] noise."""
    lower, upper = -tau, 1.0 - tau
    underage, overage = tau, 1.0 - tau

    def risk(delta_value: float) -> float:
        delta = np.asarray([delta_value], dtype=float)
        return float(
            underage * positive_part_expectation(delta, lower, upper)[0]
            + overage * negative_part_expectation(delta, lower, upper)[0]
        )

    return max(risk(estimate - oracle) - risk(0.0), 0.0)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict] = []
    theoretical_slope = -2.0 * args.beta / (2.0 * args.beta + 1.0)
    oracle_by_query = {
        query_point: float(
            oracle_quantile(
                np.asarray([query_point]),
                query_point,
                args.beta,
                dgp=args.dgp,
                holder_knots=args.holder_knots,
                holder_coefficient=args.holder_coefficient,
            )[0]
        )
        for query_point in args.query_points
    }

    for n in args.sample_sizes:
        bandwidth = args.bandwidth_constant * n ** (-1.0 / (2.0 * args.beta + 1.0))
        print(
            f"n={n}: bandwidth={bandwidth:.6f}, "
            f"query_points={args.query_points}, replications={args.replications}"
        )
        for replication in range(args.replications):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, n, replication]))
            x_train = rng.uniform(0.0, 1.0, size=n)
            noise = rng.uniform(0.0, 1.0, size=n) - args.tau
            demand = (
                oracle_quantile(
                    x_train,
                    args.query_point,
                    args.beta,
                    dgp=args.dgp,
                    holder_knots=args.holder_knots,
                    holder_coefficient=args.holder_coefficient,
                )
                + noise
            )
            for query_point in args.query_points:
                oracle_at_query = oracle_by_query[query_point]
                estimate, effective_n = fit_kpqr_at_query(
                    x_train=x_train,
                    demand=demand,
                    query_point=query_point,
                    bandwidth=bandwidth,
                    degree=args.degree,
                    tau=args.tau,
                    kernel=args.kernel,
                )
                excess_risk = exact_excess_risk(estimate, oracle_at_query, args.tau)
                records.append(
                    {
                        "sample_size": n,
                        "query_point": query_point,
                        "replication": replication,
                        "seed_entropy": f"{args.seed}:{n}:{replication}:{query_point}",
                        "bandwidth": bandwidth,
                        "effective_n": effective_n,
                        "oracle_decision": oracle_at_query,
                        "estimated_decision": estimate,
                        "estimation_error": estimate - oracle_at_query,
                        "excess_risk": excess_risk,
                        "theoretical_slope": theoretical_slope,
                    }
                )
    return pd.DataFrame(records)


def summarize_results(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for (query_point, n), group in raw_df.groupby(["query_point", "sample_size"], sort=True):
        values = group["excess_risk"].to_numpy(dtype=float)
        count = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        se = std / np.sqrt(count)
        critical = float(student_t.ppf(0.975, count - 1))
        rows.append(
            {
                "sample_size": int(n),
                "query_point": float(query_point),
                "replications": count,
                "bandwidth": float(group["bandwidth"].iloc[0]),
                "mean_effective_n": float(group["effective_n"].mean()),
                "mean_excess_risk": mean,
                "median_excess_risk": float(np.median(values)),
                "std_excess_risk": std,
                "se_excess_risk": se,
                "ci95_lower": max(mean - critical * se, 0.0),
                "ci95_upper": mean + critical * se,
            }
        )
    summary_df = pd.DataFrame(rows)
    if np.any(summary_df["mean_excess_risk"] <= 0):
        raise ValueError("Mean excess risk must be positive for the log-log regression.")

    fit_rows: list[dict] = []
    theoretical_slope = float(raw_df["theoretical_slope"].iloc[0])
    for query_point, group in summary_df.groupby("query_point", sort=True):
        log_n = np.log(group["sample_size"].to_numpy(dtype=float))
        log_risk = np.log(group["mean_excess_risk"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(log_n, log_risk, deg=1)
        fitted = intercept + slope * log_n
        residual_sum = float(np.sum((log_risk - fitted) ** 2))
        total_sum = float(np.sum((log_risk - np.mean(log_risk)) ** 2))
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
        fit_rows.append(
            {
                "query_point": float(query_point),
                "estimated_slope": float(slope),
                "intercept": float(intercept),
                "r_squared": r_squared,
                "theoretical_slope": theoretical_slope,
                "slope_difference": float(slope - theoretical_slope),
            }
        )
    fit_df = pd.DataFrame(fit_rows)
    return summary_df, fit_df


def plot_results(summary_df: pd.DataFrame, fit_df: pd.DataFrame, output_base: Path) -> None:
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
    query_points = list(summary_df.groupby("query_point", sort=True).groups.keys())
    n_panels = len(query_points)
    if n_panels == 1:
        fig, axes = plt.subplots(figsize=(7.5, 5.1))
        axes = np.asarray([axes])
    else:
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(5.15 * n_panels, 4.95),
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes)
    colors = ["#1F4E79", "#C44E52", "#55A868", "#8172B3", "#CCB974"]
    for idx, (query_point, group) in enumerate(summary_df.groupby("query_point", sort=True)):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        n = group["sample_size"].to_numpy(dtype=float)
        risk = group["mean_excess_risk"].to_numpy(dtype=float)
        lower = group["ci95_lower"].to_numpy(dtype=float)
        upper = group["ci95_upper"].to_numpy(dtype=float)
        fit = fit_df.loc[np.isclose(fit_df["query_point"], query_point)].iloc[0]
        fitted = np.exp(fit["intercept"]) * n ** fit["estimated_slope"]
        anchor_index = len(n) // 2
        theoretical = risk[anchor_index] * (n / n[anchor_index]) ** fit["theoretical_slope"]
        ax.fill_between(n, np.maximum(lower, np.finfo(float).tiny), upper, color=color, alpha=0.08)
        ax.plot(
            n,
            risk,
            "o-",
            color=color,
            linewidth=2.0,
            markersize=6.0,
            label="Mean excess risk",
        )
        ax.plot(
            n,
            fitted,
            "--",
            color=color,
            linewidth=1.4,
            alpha=0.9,
            label=rf"Fit (slope={fit['estimated_slope']:.3f})",
        )
        ax.plot(
            n,
            theoretical,
            ":",
            color="#222222",
            linewidth=1.8,
            label=f"Theory (slope={fit['theoretical_slope']:.3f})",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training sample size $n$")
        if idx == 0:
            ax.set_ylabel("Conditional excess risk")
        ax.set_title(rf"$x_0={query_point:.2f}$")
        ax.legend(frameon=False, loc="upper right")
        ax.text(
            0.05,
            0.06,
            rf"$R^2$ = {fit['r_squared']:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB"},
        )
    fig.tight_layout(w_pad=1.1)
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    timestamp = args.output_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    theoretical_slope = -2.0 * args.beta / (2.0 * args.beta + 1.0)
    print("KPQR log-log rate validation")
    print(f"beta={args.beta}, d=1, degree={args.degree}, tau={args.tau}")
    print(f"DGP={args.dgp}, query_points={args.query_points}")
    print(f"Theoretical slope: {theoretical_slope:.6f}")
    raw_df = run_experiment(args)
    summary_df, fit_df = summarize_results(raw_df)

    raw_path = DATA_DIR / f"rate_validation_raw_{timestamp}.csv"
    summary_path = DATA_DIR / f"rate_validation_summary_{timestamp}.csv"
    fit_path = DATA_DIR / f"rate_validation_slope_{timestamp}.csv"
    metadata_path = DATA_DIR / f"rate_validation_metadata_{timestamp}.json"
    figure_tag = timestamp
    figure_base = FIGURE_DIR / f"LogLog_Rate_Validation_{figure_tag}"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    fit_df.to_csv(fit_path, index=False)

    metadata = {
        "experiment": "contextwise_kpqr_excess_risk_rate",
        "sample_sizes": args.sample_sizes,
        "replications": args.replications,
        "dimension": 1,
        "beta": args.beta,
        "polynomial_degree": args.degree,
        "tau": args.tau,
        "query_points": args.query_points,
        "dgp": args.dgp,
        "holder_knots": args.holder_knots,
        "holder_coefficient": args.holder_coefficient,
        "bandwidth_constant": args.bandwidth_constant,
        "bandwidth_rule": "c * n^(-1/(2*beta+d))",
        "kernel": args.kernel,
        "seed": args.seed,
        "slope_fit": fit_df.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    plot_results(summary_df, fit_df, figure_base)

    print(summary_df.to_string(index=False))
    print(fit_df.to_string(index=False))
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Slope fit: {fit_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Figure: {figure_base.with_suffix('.pdf')}")
    print(f"Figure: {figure_base.with_suffix('.png')}")
    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
