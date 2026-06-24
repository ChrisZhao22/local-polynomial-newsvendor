"""Fixed-split CVXPY linear newsvendor benchmarks."""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import pandas as pd

from benchmark_protocol import load_fixed_split, newsvendor_cost
from runtime_logger import record_runtime
from simulation_paths import scenario_csv_path


LAMBDA_CANDIDATES = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]


def _fit_linear_newsvendor(x_train, y_train, tau, lambda_value, penalty):
    feature_dim = x_train.shape[1]
    beta_0 = cp.Variable(1)
    beta = cp.Variable(feature_dim)
    predictions = beta_0 + x_train @ beta
    loss = cp.sum(
        tau * cp.maximum(y_train - predictions, 0)
        + (1.0 - tau) * cp.maximum(predictions - y_train, 0)
    ) / len(y_train)

    if penalty == "l1":
        regularization = lambda_value * cp.norm(beta, 1)
    elif penalty == "l2":
        regularization = lambda_value * cp.sum_squares(beta)
    else:
        regularization = 0.0

    # Coefficients are intentionally unconstrained: negative feature effects
    # are present in Scenario B and must be learnable by every linear baseline.
    problem = cp.Problem(cp.Minimize(loss + regularization))
    try:
        problem.solve(solver=cp.GUROBI)
    except cp.SolverError:
        problem.solve()
    if beta_0.value is None or beta.value is None:
        raise RuntimeError(f"Linear {penalty} optimization failed with status={problem.status}.")
    return float(beta_0.value[0]), np.asarray(beta.value, dtype=float)


def run_linear_benchmark(penalty: str, model_name: str, output_stem: str):
    data = load_fixed_split()
    config = data.config
    tau = float(config.get("tau", 0.75))
    scale_factor = float(np.max(np.sum(np.abs(data.x_train), axis=1)))
    scale_factor = scale_factor if scale_factor > 0 else 1.0
    x_train = data.x_train / scale_factor
    x_validation = data.x_validation / scale_factor
    x_test = data.x_test / scale_factor

    candidates = [0.0] if penalty == "none" else LAMBDA_CANDIDATES
    selection_records = []
    fitted = {}
    for lambda_value in candidates:
        intercept, coefficients = _fit_linear_newsvendor(
            x_train, data.y_train, tau, lambda_value, penalty
        )
        fitted[lambda_value] = (intercept, coefficients)
        validation_decision = intercept + x_validation @ coefficients
        validation_cost = newsvendor_cost(validation_decision, data.y_validation, tau)
        selection_records.append(
            {
                "penalty": penalty,
                "lambda_value": lambda_value,
                "validation_mean_cost": float(np.mean(validation_cost)),
            }
        )

    best = min(selection_records, key=lambda row: row["validation_mean_cost"])
    best_lambda = float(best["lambda_value"])
    intercept, coefficients = fitted[best_lambda]

    test_start = time.time()
    decision = intercept + x_test @ coefficients
    operation_cost = newsvendor_cost(decision, data.y_test, tau)
    test_runtime = time.time() - test_start
    record_runtime(model_name, test_runtime, len(data.y_test), config_file="benchmark_config.json")

    output = pd.DataFrame(
        {
            "Selected_Lambda": np.full(len(data.y_test), best_lambda),
            "Decision_Q": decision,
            "Demand_D": data.y_test,
            "Operation_Cost": operation_cost,
        }
    )
    output["Coef_Intercept"] = intercept
    for index, value in enumerate(coefficients, start=1):
        output[f"Coef_Feat_{index}"] = value

    output_file = scenario_csv_path(f"{output_stem}.csv", config)
    selection_file = scenario_csv_path(f"{output_stem}_lambda_selection.csv", config)
    output.to_csv(output_file, index=False)
    selection_df = pd.DataFrame(selection_records)
    selection_df["is_best"] = selection_df["lambda_value"] == best_lambda
    selection_df.to_csv(selection_file, index=False)
    print(
        f"{model_name}: lambda={best_lambda:g}, validation={best['validation_mean_cost']:.6f}, "
        f"test={np.mean(operation_cost):.6f}"
    )
