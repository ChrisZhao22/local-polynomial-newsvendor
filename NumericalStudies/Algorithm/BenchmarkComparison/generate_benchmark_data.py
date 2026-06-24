import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
from scipy.stats import norm
from simulation_paths import scenario_csv_path

DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "benchmark_config.json")

# ==========================================
# Unified Configuration Defaults
# ==========================================
TOTAL_LEN = 5000
LNTR = 3000
LNVA = 1000
LNTE = 1000
TAU = 0.75
OUTPUT_DIR = "../../Data/BenchmarkComparison"

# 场景选择: 'a' (MA), 'b' (MC)
DEFAULT_SCENARIO = "b"
DEFAULT_SEED = 2026
DEFAULT_NOISE_SCALE = 0.25


def generate_independent_uniform_features(n, dim):
    return np.random.uniform(0, 1, size=(n, dim))


def generate_scenario_a_ma(n):
    """
    Scenario a: Multivariate Additive (MA)
    f(x) = exp(x1 - 0.5) + 2(x2 + x3 - 1)^2 + |x4 - 0.5|
    """
    print("Generating Scenario A: Multivariate Additive (MA)...")
    X = generate_independent_uniform_features(n, dim=4)
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    f_x = np.exp(x1 - 0.5) + 2 * ((x2 + x3 - 1) ** 2) + np.abs(x4 - 0.5)
    return X, f_x


def generate_scenario_b_mc(n):
    """
    Scenario b: MC (Moderate Non-linear, 8D)
    """
    print("Generating Scenario B: MC (Moderate Non-linear, 8D)...")
    X = generate_independent_uniform_features(n, dim=8)
    x1, x2, x3, x4, x5, x6, x7, x8 = [X[:, i] for i in range(8)]
    f_x = (
        2.0
        + 1.5 * x1
        + 2.5 * (x2**2)
        + 3.0 * np.sin(np.pi * x3)
        + 1.2 * (x4 * x5)
        + 0.8 * np.exp(x6)
        - 1.5 * x7
        + 2.0 * np.log(1 + x8)
    )
    f_x = np.maximum(f_x, 0.05)
    return X, f_x


SCENARIOS = {
    "a": generate_scenario_a_ma,
    "b": generate_scenario_b_mc,
}


def _load_generation_params_from_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        "scenario": str(config.get("scenario", DEFAULT_SCENARIO)).strip().lower(),
        "seed": int(config.get("seed", DEFAULT_SEED)),
        "noise_scale": float(config.get("noise_scale", DEFAULT_NOISE_SCALE)),
        "total_len": int(config.get("total_len", TOTAL_LEN)),
        "lntr": int(config.get("lntr", LNTR)),
        "lnva": int(config.get("lnva", LNVA)),
        "lnte": int(config.get("lnte", LNTE)),
        "tau": float(config.get("tau", TAU)),
    }


GENERATION_CONFIG = _load_generation_params_from_config()
CURRENT_SCENARIO = GENERATION_CONFIG["scenario"]
SEED = GENERATION_CONFIG["seed"]
NOISE_SCALE = GENERATION_CONFIG["noise_scale"]


def generate_quantile_centered_noise(sigma_x, tau):
    raw_noise = np.random.normal(loc=0.0, scale=sigma_x, size=len(sigma_x))
    tau_shift = norm.ppf(tau, loc=0.0, scale=sigma_x)
    return raw_noise - tau_shift


def run_data_generation(
    current_scenario,
    noise_scale,
    seed,
    total_len=TOTAL_LEN,
    lntr=LNTR,
    lnva=LNVA,
    lnte=LNTE,
    tau=TAU,
    output_dir=OUTPUT_DIR,
    csv_name=None,
    config_name="benchmark_config.json",
):
    if current_scenario not in SCENARIOS:
        raise ValueError(f"Invalid scenario '{current_scenario}'.")
    if lntr + lnva + lnte != total_len:
        raise ValueError(
            "Fixed benchmark protocol requires lntr + lnva + lnte == total_len; "
            f"got {lntr} + {lnva} + {lnte} != {total_len}."
        )

    np.random.seed(seed)
    output_dir_abs = output_dir if os.path.isabs(output_dir) else os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    generator_func = SCENARIOS[current_scenario]
    X, f_x = generator_func(total_len)

    sigma_x = np.full(total_len, noise_scale)
    noise = generate_quantile_centered_noise(sigma_x, tau)
    demand = f_x + noise

    negative_count = np.sum(demand < 0)
    print(f"Original Min Demand: {demand.min():.4f}")
    print(f"Samples truncated to 0: {negative_count} ({negative_count / total_len:.1%})")
    demand = np.maximum(demand, 0)

    num_features = X.shape[1]
    feature_cols = [f"Feature_{i + 1}" for i in range(num_features)]
    columns = feature_cols + ["Demand", "Optimal_decision"]
    data_matrix = np.hstack([X, demand.reshape(-1, 1), f_x.reshape(-1, 1)])
    df = pd.DataFrame(data_matrix, columns=columns)

    if csv_name is None:
        csv_filename = scenario_csv_path(
            "newsvendor_simulation_data.csv",
            current_scenario,
            output_dir_abs,
        )
    else:
        csv_filename = os.path.join(output_dir_abs, csv_name)
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to: {csv_filename}")
    print(f"Shape: {df.shape}")

    config = {
        "total_len": total_len,
        "lntr": lntr,
        "lnva": lnva,
        "lnte": lnte,
        "scenario": current_scenario,
        "noise_scale": noise_scale,
        "seed": seed,
        "tau": tau,
    }

    if os.path.isabs(config_name):
        json_filename = config_name
    else:
        json_filename = os.path.join(SCRIPT_DIR, config_name)
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to: {json_filename}")

    return df, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate the benchmark simulation dataset specified by "
            "benchmark_config.json."
        )
    )
    parser.parse_args()
    run_data_generation(
        current_scenario=CURRENT_SCENARIO,
        noise_scale=NOISE_SCALE,
        seed=SEED,
        total_len=GENERATION_CONFIG["total_len"],
        lntr=GENERATION_CONFIG["lntr"],
        lnva=GENERATION_CONFIG["lnva"],
        lnte=GENERATION_CONFIG["lnte"],
        tau=GENERATION_CONFIG["tau"],
        csv_name=None,
        config_name="benchmark_config.json",
    )
