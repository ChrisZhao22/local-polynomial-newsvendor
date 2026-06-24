import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import pandas as pd
import json
from scipy.stats import norm
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
Feature_Raw, feature_cols = load_feature_matrix(df, config)
lntr = config["lntr"]
lnva = config["lnva"]
lnte = config["lnte"]
function_scenario = config["scenario"]
TOTAL_LEN = len(Demand)
print(f"已加载数据. Features shape: {Feature_Raw.shape}")

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
# 4. 主循环逻辑
# ==========================================
# 初始化
operation_cost = np.zeros(lnte)
muD = np.zeros(lnte)
sigmaD = np.zeros(lnte)
Q_pred = np.zeros(lnte)

# 记录系数 (Intercept + 2个特征)
coef_history = np.zeros((lnte, 1 + Feature_Raw.shape[1]))

start_time = time.time()
start_idx = lntr + lnva
X_train_raw = Feature_Raw[:lntr, :]
scale_factor = np.max(np.sum(np.abs(X_train_raw), axis=1))
scale_factor = scale_factor if scale_factor > 0 else 1.0
X_train = X_train_raw / scale_factor
y_train = Demand[:lntr]

model_mu = LinearRegression(fit_intercept=True)
model_mu.fit(X_train, y_train)
residuals = y_train - model_mu.predict(X_train)
y_train_var = np.log(residuals**2 + 1e-8)
model_sigma = LinearRegression(fit_intercept=True)
model_sigma.fit(X_train, y_train_var)

X_test = Feature_Raw[start_idx : start_idx + lnte] / scale_factor
muD[:] = model_mu.predict(X_test)
sigmaD[:] = np.exp(model_sigma.predict(X_test) / 2)
Q_pred[:] = np.maximum(muD + sigmaD * norm.ppf(r), 0.0)
operation_cost[:] = nv_cost(Q_pred, Demand[start_idx : start_idx + lnte], b, h)
coef_history[:, 0] = model_mu.intercept_
coef_history[:, 1:] = model_mu.coef_

test_runtime_sec = time.time() - start_time
print(f"Loop finished in {test_runtime_sec:.2f} s")
record_runtime("Est-Opt (OLS)", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = scenario_csv_path("nv_ETO.csv", config)

# 基础结果
data_dict = {
    "Decision_Q": Q_pred,
    "Demand_D": Demand[start_idx + delay : start_idx + lnte + delay],
    "Operation_Cost": operation_cost,
    "Mu_Pred": muD,
    "Sigma_Pred": sigmaD,
}
# 添加系数列
for idx in range(coef_history.shape[1]):
    col_name = "Beta_Intercept" if idx == 0 else f"Beta_Feat_{idx}"
    data_dict[col_name] = coef_history[:, idx]

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")
