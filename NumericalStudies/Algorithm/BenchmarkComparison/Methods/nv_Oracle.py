import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import pandas as pd
import json
import time
from runtime_logger import record_runtime
from simulation_paths import scenario_csv_path

# ==========================================
# 1. 数据加载
# ==========================================
config_file = "benchmark_config.json"

# 读取配置
with open(config_file, "r") as f:
    config = json.load(f)

data_file = scenario_csv_path("newsvendor_simulation_data.csv", config)
df = pd.read_csv(data_file)
Demand = df["Demand"].values
Optimal_decision = df["Optimal_decision"].values

lntr = config["lntr"]
lnva = config["lnva"]
lnte = config["lnte"]
function_scenario = config["scenario"]
delay = 0
b = 3 / 4
h = 1 / 4
alpha = b / (b + h)  # Target Quantile alpha
n = lntr


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


Q_pred = np.zeros(lnte)
operation_cost = np.zeros(lnte)
start_idx = lntr + lnva

# 求解主循环
test_start_time = time.time()
for k in range(lnte):
    t = start_idx + k
    actual_demand = Demand[t + delay]
    Q_pred[k] = Optimal_decision[t + delay]
    operation_cost[k] = nv_cost(Q_pred[k], actual_demand, b, h)
test_runtime_sec = time.time() - test_start_time
print(f"Test loop finished in {test_runtime_sec:.4f} seconds")
record_runtime("Oracle", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = scenario_csv_path("nv_Oracle.csv", config)

df_out = pd.DataFrame(
    {
        "Decision_Q": Q_pred,
        "Demand_D": Demand[start_idx : start_idx + lnte],
        "Operation_Cost": operation_cost,
    }
)

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")
