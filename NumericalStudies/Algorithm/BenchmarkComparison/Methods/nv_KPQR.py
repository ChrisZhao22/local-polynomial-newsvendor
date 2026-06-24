import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import pandas as pd
import json
import time
import cvxpy as cp
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from feature_loader import load_feature_matrix
from runtime_logger import record_runtime
from simulation_paths import scenario_csv_path

# ==========================================
# 1. 数据加载
# ==========================================
config_file = "benchmark_config.json"

# 读取配置
with open(config_file, "r") as f:
    config = json.load(f)

# 读取当前场景对应的 CSV
data_file = scenario_csv_path("newsvendor_simulation_data.csv", config)
df = pd.read_csv(data_file)
Demand = df["Demand"].values
Features_Raw, feature_cols = load_feature_matrix(df, config)

lntr = config["lntr"]
lnva = config["lnva"]
lnte = config["lnte"]
function_scenario = config["scenario"]
TOTAL_LEN = len(Demand)

print(f"已加载数据: {data_file}. Total Samples: {TOTAL_LEN}")
print(f"   Features Shape: {Features_Raw.shape}")

# ==========================================
# 2. 参数设置
# ==========================================
# 报童参数
delay = 0
b = 3 / 4
h = 1 / 4
alpha = b / (b + h)  # Target Quantile

# KPQR 模型参数
# k = floor(beta)。多项式阶数 k.
poly_degree = 2
# 候选带宽列表（先在验证集选优，再固定到测试集）
bandvec = [0.20, 0.30, 0.40, 0.50, 0.60]

print(f"KPQR Config: Degree={poly_degree}, Candidate Bandwidths={bandvec}")


# ==========================================
# 3. 辅助函数：KPQR LP 求解器
# ==========================================
def solve_kpqr_lp(X_train, Y_train, x_query, bandwidth, degree, b, h):
    """
    求解 KPQR 的线性规划问题
    """
    n_samples, n_features = X_train.shape

    # 1. 计算距离与核权重 (Gaussian Kernel)
    # 使用欧氏距离 (Euclidean Distance)
    dists = np.linalg.norm(X_train - x_query, axis=1)

    # K_h(u) = 1/h * K(u/h). 这里计算相对权重，常数因子不影响 LP 最优解
    # 避免数值下溢，截断极小权重
    weights = norm.pdf(dists / bandwidth)

    # 筛选有效样本 (Effective Samples) 以加速 LP
    # 仅保留权重显著的样本 (例如 > 1e-6)
    mask = weights > 1e-6
    effective_n = int(np.sum(mask))
    if effective_n < (n_features * degree + 5):  # 确保有足够样本求解
        # 如果有效样本太少，退化为全局分位数
        return np.quantile(Y_train, b / (b + h)), effective_n

    X_eff = X_train[mask]
    Y_eff = Y_train[mask]
    W_eff = weights[mask]

    # 2. 局部中心化与多项式扩展
    # 将 x_query 映射为原点，这样求解出的截距项即为预测值 f(x_query)
    # Taylor 展开: f(x) approx f(x_0) + f'(x_0)(x-x_0) + ...
    X_centered = (X_eff - x_query) / bandwidth  # 缩放增加数值稳定性

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_design = poly.fit_transform(X_centered)

    num_params = X_design.shape[1]
    n_eff = len(Y_eff)

    # 3. CVXPY 线性规划建模
    # Min sum w_i * (b * u_i + h * v_i)
    # s.t. Y - X*theta = u - v, u>=0, v>=0

    theta = cp.Variable(num_params)
    u_plus = cp.Variable(n_eff)
    u_minus = cp.Variable(n_eff)

    # 目标函数：加权 Newsvendor Cost (Check Loss)
    # 注意：Check Loss = u * (alpha - I(u<0))
    # Newsvendor Cost = b*(d-q)+ + h*(q-d)+
    # 两者是等价的，这里直接用 b, h 表达
    weighted_loss = cp.sum(cp.multiply(W_eff, b * u_plus + h * u_minus))

    objective = cp.Minimize(weighted_loss)

    constraints = [Y_eff == X_design @ theta + u_plus - u_minus, u_plus >= 0, u_minus >= 0]

    prob = cp.Problem(objective, constraints)

    try:
        # 使用 GUROBI 或 OSQP 求解
        prob.solve(solver=cp.GUROBI, verbose=False)
    except:
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except:
            prob.solve(verbose=False)

    # 4. 返回结果
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        # 截距项即为 x_query 处的预测分位数 (因为我们做了中心化)
        return theta.value[0], effective_n
    else:
        # 求解失败 fallback
        return np.quantile(Y_eff, b / (b + h)), effective_n


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环 (KPQR Estimation)
# ==========================================
def run_kpqr_rollout(start_idx, horizon, bandwidth):
    Q_pred = np.zeros(horizon)
    operation_cost = np.zeros(horizon)
    # 用于记录有效样本数，观察带宽是否合理
    effective_n = np.zeros(horizon)
    demand_eval = Demand[start_idx + delay : start_idx + horizon + delay]

    for k in range(horizon):
        t = start_idx + k

        if k % 50 == 0:
            print(f"Step {k}/{horizon}")

        # A. 准备数据 (滑动窗口)
        # 使用与 nv_KO.py 相同的归一化策略，保证距离度量的一致性
        window_features = np.vstack([Features_Raw[:lntr, :], Features_Raw[t, :]]).astype(float)

        # 归一化: 除以绝对值最大值 (MaxAbsScaler风格)
        scale_factor = np.max(np.abs(window_features), axis=0)
        # 防止除零
        scale_factor[scale_factor == 0] = 1.0

        FeaturesT = window_features / scale_factor

        # 分离历史数据与当前测试点
        X_train = FeaturesT[:-1, :]
        x_eval = FeaturesT[-1, :].reshape(1, -1)  # 保持二维

        Y_train = Demand[:lntr]

        # B. 调用 KPQR 求解
        # 注意：x_eval 传入 flatten 形式方便计算距离
        q_opt, eff_n = solve_kpqr_lp(X_train, Y_train, x_eval.flatten(), bandwidth, poly_degree, b, h)

        # 非负截断
        q_opt = max(0, q_opt)

        # C. 记录与评估
        Q_pred[k] = q_opt
        effective_n[k] = eff_n

        if t + delay < TOTAL_LEN:
            actual_demand = Demand[t + delay]
            operation_cost[k] = nv_cost(q_opt, actual_demand, b, h)

    return Q_pred, operation_cost, effective_n, demand_eval


selection_records = []
overall_start_time = time.time()

val_start_idx = lntr
for bandwidth in bandvec:
    print(f"Validating KPQR bandwidth: {bandwidth}")
    _, val_costs, val_effective_n, _ = run_kpqr_rollout(val_start_idx, lnva, bandwidth)
    val_mean_cost = float(np.mean(val_costs))
    val_mean_effective_n = float(np.mean(val_effective_n))
    selection_records.append(
        {
            "bandwidth": bandwidth,
            "validation_mean_cost": val_mean_cost,
            "validation_mean_effective_n": val_mean_effective_n,
        }
    )
    print(
        f"Validation mean cost (bw={bandwidth}): {val_mean_cost:.6f}, "
        f"mean effective samples: {val_mean_effective_n:.2f}"
    )

best_record = min(selection_records, key=lambda x: x["validation_mean_cost"])
best_bandwidth = best_record["bandwidth"]
best_val_cost = best_record["validation_mean_cost"]

print(f"Selected best bandwidth: {best_bandwidth} (validation mean cost = {best_val_cost:.6f})")

test_start_idx = lntr + lnva
print(f"Start KPQR test evaluation (Bandwidth={best_bandwidth}, Degree={poly_degree})...")
test_start_time = time.time()
Q_pred, operation_cost, Effective_N, Demand_eval = run_kpqr_rollout(test_start_idx, lnte, best_bandwidth)
test_runtime_sec = time.time() - test_start_time
test_mean_cost = float(np.mean(operation_cost))

end_time = time.time()
print(f"Total time: {end_time - overall_start_time:.2f} seconds")
print(f"Test runtime: {test_runtime_sec:.2f} seconds")
print(f"Test mean cost with selected bandwidth: {test_mean_cost:.6f}")
record_runtime("KPQR", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = scenario_csv_path("nv_KPQR.csv", config)
selection_filename = scenario_csv_path("nv_KPQR_bandwidth_selection.csv", config)

df_out = pd.DataFrame(
    {
        "Selected_Bandwidth": np.full(lnte, best_bandwidth),
        "Decision_Q": Q_pred,
        "Demand_D": Demand_eval,
        "Operation_Cost": operation_cost,
        "Effective_N": Effective_N,
    }
)

df_out.to_csv(output_filename, index=False)
df_selection = pd.DataFrame(selection_records)
df_selection["is_best"] = df_selection["bandwidth"] == best_bandwidth
df_selection.to_csv(selection_filename, index=False)
print(f"Results saved to {output_filename}")
print(f"Bandwidth selection summary saved to {selection_filename}")
