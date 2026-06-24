import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import time
from feature_loader import load_feature_matrix
from runtime_logger import record_runtime
from simulation_paths import scenario_csv_path

# ==========================================
# 1. 数据加载
# ==========================================
config_file = "benchmark_config.json"

with open(config_file, "r") as f:
    config = json.load(f)
data_file = scenario_csv_path("newsvendor_simulation_data.csv", config)
df = pd.read_csv(data_file)
Demand = df["Demand"].values
Features_Raw, feature_cols = load_feature_matrix(df, config)
lF = Features_Raw.shape[1]
lntr = config["lntr"]
lnva = config["lnva"]
lnte = config["lnte"]
function_scenario = config["scenario"]
TOTAL_LEN = len(Demand)

print(f"已加载数据. Features: {lF}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 0
b = 3 / 4
h = 1 / 4
r = b / (b + h)


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环
# ==========================================
operation_cost = np.zeros(lnte)
Q_pred = np.zeros(lnte)
muD = np.zeros(lnte)
sigmaD = np.zeros(lnte)
ResOpt = np.zeros(lnte)

print("Processing Scarf Rule...")
start_time = time.time()
start_idx = lntr + lnva
X_train_raw = Features_Raw[:lntr, :]
scale_factor = np.max(np.sum(np.abs(X_train_raw), axis=1))
scale_factor = scale_factor if scale_factor > 0 else 1.0
X_train = X_train_raw / scale_factor
y_train = Demand[:lntr]

model_mean = LinearRegression(fit_intercept=True)
model_mean.fit(X_train, y_train)
residuals = y_train - model_mean.predict(X_train)
model_var = LinearRegression(fit_intercept=True)
model_var.fit(X_train, np.log(residuals**2 + 1e-8))

X_test = Features_Raw[start_idx : start_idx + lnte] / scale_factor
muD[:] = model_mean.predict(X_test)
sigmaD[:] = np.exp(model_var.predict(X_test) / 2)
ResOpt[:] = np.sum(residuals**2)
scarf_term = np.sqrt(b / h) - np.sqrt(h / b)
Q_pred[:] = muD + (sigmaD / 2.0) * scarf_term
operation_cost[:] = nv_cost(Q_pred, Demand[start_idx : start_idx + lnte], b, h)

test_runtime_sec = time.time() - start_time
print(f"Finished in {test_runtime_sec:.2f}s")
record_runtime("Minimax (Scarf)", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = scenario_csv_path("nv_scarf.csv", config)

data_dict = {
    "Decision_Q": Q_pred,
    "Demand_D": Demand[start_idx + delay : start_idx + lnte + delay],
    "Operation_Cost": operation_cost,
    "Mu_Pred": muD,
    "Sigma_Pred": sigmaD,
    "Residual_Sum": ResOpt,
}

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Saved {output_filename}")

