import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import pandas as pd
import json
from scipy.stats import norm
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
lntr = config["lntr"]
lnva = config["lnva"]
lnte = config["lnte"]
function_scenario = config["scenario"]
TOTAL_LEN = len(Demand)

print(f"已加载数据. Features shape: {Features_Raw.shape}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 0
bandvec = [0.06, 0.08, 0.10, 0.12, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]  # 候选带宽列表（先在验证集选优）
b = 3 / 4
h = 1 / 4
r = b / (b + h)


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 5. 核心滚动评估逻辑
# ==========================================
def run_kernel_rollout(start_idx, horizon, bandwidth):
    Q_pred = np.zeros(horizon)
    Cost_list = np.zeros(horizon)
    demand_eval = Demand[start_idx + delay : start_idx + horizon + delay]

    for k in range(horizon):
        t = start_idx + k

        # A. 特征归一化(严格遵循gahyiban操作，滑动标准化+绝对值最大)
        window_features = np.vstack([Features_Raw[:lntr, :], Features_Raw[t, :]]).astype(float)
        scale_factor = np.max(np.abs(window_features), axis=0)
        scale_factor[scale_factor == 0] = 1.0
        FeaturesT = window_features / scale_factor

        current_feat = FeaturesT[-1, :]
        history_feats = FeaturesT[:-1, :]

        # B. 核权重
        dists = np.linalg.norm(history_feats - current_feat, axis=1)
        weights = norm.pdf(dists / bandwidth)
        weight_sum = np.sum(weights)
        weights_norm = weights / (weight_sum if weight_sum > 0 else 1.0)

        demand_h = Demand[:lntr]

        sort_idx = np.argsort(demand_h)  # 给出demand从小到大排序后的索引
        sDemand = demand_h[sort_idx]  # 得到真正从小到大排序后的需求量
        sWeights = weights_norm[sort_idx]  # 得到真正从小到大排序后的需求量对应的权重

        kernel_cdf = np.cumsum(sWeights)
        idx_candidates = np.flatnonzero(kernel_cdf >= r)
        idx_opt = idx_candidates[0] if idx_candidates.size > 0 else len(sDemand) - 1
        q0 = sDemand[idx_opt]

        # D. 记录
        Q_pred[k] = q0
        actual = Demand[t + delay]
        Cost_list[k] = nv_cost(q0, actual, b, h)

    return Q_pred, Cost_list, demand_eval


# ==========================================
# 6. 先在验证集选带宽，再在测试集评估
# ==========================================
results_dict = {}
selection_records = []

overall_start_time = time.time()

val_start_idx = lntr
for bandwidth in bandvec:
    print(f"Validating bandwidth: {bandwidth}")
    _, val_costs, _ = run_kernel_rollout(val_start_idx, lnva, bandwidth)
    val_mean_cost = float(np.mean(val_costs))
    selection_records.append({"bandwidth": bandwidth, "validation_mean_cost": val_mean_cost})
    print(f"Validation mean cost (bw={bandwidth}): {val_mean_cost:.6f}")

best_record = min(selection_records, key=lambda x: x["validation_mean_cost"])
best_bandwidth = best_record["bandwidth"]
best_val_cost = best_record["validation_mean_cost"]

print(f"Selected best bandwidth: {best_bandwidth} (validation mean cost = {best_val_cost:.6f})")

test_start_idx = lntr + lnva
test_start_time = time.time()
Q_pred, Cost_list, Demand_eval = run_kernel_rollout(test_start_idx, lnte, best_bandwidth)
test_runtime_sec = time.time() - test_start_time
test_mean_cost = float(np.mean(Cost_list))

results_dict["Selected_Bandwidth"] = np.full(lnte, best_bandwidth)
results_dict["Decision_Q"] = Q_pred
results_dict["Demand_D"] = Demand_eval
results_dict["Operation_Cost"] = Cost_list

print(f"Total time: {time.time() - overall_start_time:.4f} s")
print(f"Test runtime: {test_runtime_sec:.4f} s")
print(f"Test mean cost with selected bandwidth: {test_mean_cost:.6f}")
record_runtime("KO", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 7. 保存结果 (CSV)
# ==========================================
output_filename = scenario_csv_path("nv_KO.csv", config)
# selection_filename = scenario_csv_path("nv_KO_bandwidth_selection.csv", config)

df_out = pd.DataFrame(results_dict)
df_out.to_csv(output_filename, index=False)
# df_selection = pd.DataFrame(selection_records)
# df_selection["is_best"] = df_selection["bandwidth"] == best_bandwidth
# df_selection.to_csv(selection_filename, index=False)
print(f"Results saved to {output_filename}")
# print(f"Bandwidth selection summary saved to {selection_filename}")

