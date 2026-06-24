# local-polynomial-newsvendor

This repository contains the code, data, and figures for Section 6: Numerical Studies of the paper. The experiments are organized into three parts:

1. `RateValidation`: validation of the convergence rate predicted by the theory using log-log plots.
2. `PolynomialHierarchy`: comparison of degree-0, degree-1, and degree-2 KPQR policies.
3. `BenchmarkComparison`: benchmark comparison against alternative newsvendor methods.

The directory structure is aligned across code, data, and figures:

```text
NumericalStudies/
├── Algorithm/
│   ├── RateValidation/
│   ├── PolynomialHierarchy/
│   └── BenchmarkComparison/
├── Data/
│   ├── RateValidation/
│   ├── PolynomialHierarchy/
│   └── BenchmarkComparison/
└── Figure/
    ├── RateValidation/
    ├── PolynomialHierarchy/
    └── BenchmarkComparison/
```

## Requirements

The experiments use Python and standard scientific-computing packages. The current code base was checked with the following environment:

| Package | Version |
|---|---:|
| Python | 3.13.5 |
| `numpy` | 2.1.3 |
| `pandas` | 2.2.3 |
| `matplotlib` | 3.10.0 |
| `scipy` | 1.15.3 |
| `scikit-learn` | 1.6.1 |
| `gurobipy` | 13.0.0 |
| Gurobi Optimizer | 13.0.0 |

The benchmark comparison also requires `cvxpy` for optimization-based methods and `torch` for the DNN benchmark. Several methods are modeled with `cvxpy`; when available, Gurobi is used as the preferred solver, otherwise the scripts fall back to the default `cvxpy` solver.

## 6.2 Rate Validation

The rate-validation experiment is implemented in:

```text
Algorithm/RateValidation/rate_validation.py
```

The configuration file is:

```text
Algorithm/RateValidation/rate_validation_config.json
```

***To reproduce the experiment, run:***

```bash
python NumericalStudies/Algorithm/RateValidation/rate_validation.py
```

The script writes numerical outputs to:

```text
Data/RateValidation/
```

and figures to:

```text
Figure/RateValidation/
```

The main figure used in the paper is:

```text
Figure/RateValidation/LogLog_Rate_Validation.pdf
```

## 6.3 Polynomial Hierarchy Beyond KO

The polynomial-hierarchy experiment is implemented in:

```text
Algorithm/PolynomialHierarchy/polynomial_hierarchy.py
```

The configuration file is:

```text
Algorithm/PolynomialHierarchy/polynomial_hierarchy_config.json
```

***To reproduce the experiment, run:***

```bash
python NumericalStudies/Algorithm/PolynomialHierarchy/polynomial_hierarchy.py
```

The script writes numerical outputs to:

```text
Data/PolynomialHierarchy/
```

and figures to:

```text
Figure/PolynomialHierarchy/
```

The main figures used in the paper are:

```text
Figure/PolynomialHierarchy/Polynomial_Hierarchy.pdf
Figure/PolynomialHierarchy/Polynomial_Hierarchy_Policies.pdf
```

## 6.4 Benchmark Comparison

The benchmark comparison is implemented in:

```text
Algorithm/BenchmarkComparison/
```

The main configuration file is:

```text
Algorithm/BenchmarkComparison/benchmark_config.json
```

The concrete benchmark methods are stored in:

```text
Algorithm/BenchmarkComparison/Methods/
```

The benchmark includes the following methods:

- `DNN`
- `Est-Opt`
- `KO`
- `KPQR`
- `LinearModel`
- `LinearModel-L1`
- `LinearModel-L2`
- `Oracle`
- `RKHS`
- `SAA`
- `Minimax (Scarf)`

### Full benchmark pipeline

***To run the complete benchmark pipeline, including data generation, all benchmark methods, aggregation, and figure generation:***

```bash
cd NumericalStudies/Algorithm/BenchmarkComparison
python run_benchmark_pipeline.py
```

By default, this runs both scenarios and 20 independent seeds:

```text
scenarios: a, b
seeds: 2026, ..., 2045
```

This can be time-consuming, especially because KPQR and several benchmark methods solve optimization problems.

### Run selected benchmark experiments

To run selected scenarios or seeds:

```bash
cd NumericalStudies/Algorithm/BenchmarkComparison
python run_benchmark_experiments.py --scenarios a --seeds 2026
```

To run selected methods only:

```bash
python run_benchmark_experiments.py \
    --scenarios a \
    --seeds 2026 \
    --algorithms nv_KPQR.py nv_DNN.py nv_RKHS.py
```

This script runs the benchmark experiments and writes model outputs. It does not generate the final aggregate figures.

### Aggregate and plot benchmark replications

After the benchmark replications have been run, aggregate the results and generate figures with:

```bash
cd NumericalStudies/Algorithm/BenchmarkComparison
python summarize_benchmark_replications.py --scenario a
python summarize_benchmark_replications.py --scenario b
```

The main benchmark figures used in the paper are stored in:

```text
Figure/BenchmarkComparison/
```

Specifically:

```text
Figure/BenchmarkComparison/Benchmark_Cost_Comparison_scenario_a.pdf
Figure/BenchmarkComparison/Benchmark_Cost_Comparison_scenario_b.pdf
```

### Single-seed summary

For debugging a single scenario/seed run, use:

```bash
cd NumericalStudies/Algorithm/BenchmarkComparison
python summarize_single_seed.py
```

This script reads the current `benchmark_config.json` and summarizes the corresponding single-run outputs. It is mainly intended for debugging and is not the main source of the paper figures.

## Caching and Reproducibility

The benchmark experiments use a resumable caching mechanism. For each scenario and seed, outputs are stored under paths such as:

```text
Data/BenchmarkComparison/scenario_a/seed_2026/
Data/BenchmarkComparison/scenario_b/seed_2026/
```

Each completed step writes a marker file, for example:

```text
.done_data_generation
.done_nv_KPQR
.done_nv_DNN
.complete
```

If a marker file already exists, the corresponding step is skipped when rerunning the benchmark pipeline. This makes it possible to resume interrupted runs without recomputing completed models.

To force a fresh run for a particular scenario and seed, remove the corresponding seed directory before rerunning the pipeline.

## Output Files

Each benchmark seed directory contains:

- the generated simulation data;
- one output CSV for each benchmark method;
- hyperparameter-selection files where applicable;
- runtime records;
- completion markers.

Computation-time records are stored in each seed directory, for example:

```text
Data/BenchmarkComparison/scenario_a/seed_2026/scenario_a/runtime_metrics_scenario_a.csv
```

After aggregation, runtime records across replications are also saved in:

```text
Data/BenchmarkComparison/summary/benchmark_runtime_replications_scenario_a.csv
Data/BenchmarkComparison/summary/benchmark_runtime_replications_scenario_b.csv
```

These runtime files are kept for reporting computational cost when needed; the benchmark scripts do not generate runtime figures.

Aggregate benchmark summaries are stored in:

```text
Data/BenchmarkComparison/summary/
```

Figures are stored in:

```text
Figure/
```
