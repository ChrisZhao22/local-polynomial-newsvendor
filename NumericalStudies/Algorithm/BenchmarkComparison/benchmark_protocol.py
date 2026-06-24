"""Shared fixed-split utilities for the Section 6.4 benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_loader import load_feature_matrix
from simulation_paths import load_config, scenario_csv_path


@dataclass(frozen=True)
class FixedSplitData:
    config: dict
    features: np.ndarray
    demand: np.ndarray
    optimal_decision: np.ndarray
    train_slice: slice
    validation_slice: slice
    test_slice: slice

    @property
    def x_train(self) -> np.ndarray:
        return self.features[self.train_slice]

    @property
    def y_train(self) -> np.ndarray:
        return self.demand[self.train_slice]

    @property
    def x_validation(self) -> np.ndarray:
        return self.features[self.validation_slice]

    @property
    def y_validation(self) -> np.ndarray:
        return self.demand[self.validation_slice]

    @property
    def x_test(self) -> np.ndarray:
        return self.features[self.test_slice]

    @property
    def y_test(self) -> np.ndarray:
        return self.demand[self.test_slice]


def load_fixed_split(config_file: str = "benchmark_config.json") -> FixedSplitData:
    config = load_config(config_file)
    data_file = scenario_csv_path("newsvendor_simulation_data.csv", config)
    df = pd.read_csv(data_file)
    features, _ = load_feature_matrix(df, config)
    demand = df["Demand"].to_numpy(dtype=float)
    optimal = df["Optimal_decision"].to_numpy(dtype=float)

    lntr = int(config["lntr"])
    lnva = int(config["lnva"])
    lnte = int(config["lnte"])
    total_len = int(config.get("total_len", lntr + lnva + lnte))
    if lntr + lnva + lnte != total_len:
        raise ValueError("Fixed split sizes must sum to total_len.")
    if len(df) < total_len:
        raise ValueError(f"Dataset has {len(df)} rows but fixed protocol requires {total_len}.")

    return FixedSplitData(
        config=config,
        features=features[:total_len],
        demand=demand[:total_len],
        optimal_decision=optimal[:total_len],
        train_slice=slice(0, lntr),
        validation_slice=slice(lntr, lntr + lnva),
        test_slice=slice(lntr + lnva, total_len),
    )


def newsvendor_cost(q, d, tau: float):
    q_arr = np.asarray(q, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    return tau * np.maximum(d_arr - q_arr, 0.0) + (1.0 - tau) * np.maximum(
        q_arr - d_arr, 0.0
    )
