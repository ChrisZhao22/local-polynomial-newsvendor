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
from sklearn.metrics.pairwise import rbf_kernel
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
b = 3 / 4
h = 1 / 4
tau = b / (b + h)  # Target Quantile

# 模型参数
# 工程版快速选参：先用较小网格做固定模型验证，再用最优参数跑测试阶段滚动逻辑
lambda_candidates = [1e-7, 1e-6, 1e-5]
sigma_candidates = [0.5, 1.0, 2.0]
update_step = 50


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


def solve_kernel_quantile_dual(K, Y, C, tau):
    """
    求解对偶 QP 问题
    minimize 0.5 * alpha^T * K * alpha - alpha^T * Y
    s.t. C(tau-1) <= alpha_i <= C*tau
         sum(alpha) = 0
    """
    m = len(Y)
    alpha = cp.Variable(m)

    # 目标函数
    # 0.5 * quad_form(alpha, K) - alpha.T @ Y
    # 注意: cp.quad_form 要求 K 是半正定 (PSD)。RBF 核矩阵通常是 PSD 的，
    # 但数值误差可能导致微小的负特征值，使用 psd_wrap 保证求解器接受。
    objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(K)) - alpha.T @ Y)

    # 约束条件
    constraints = [alpha >= C * (tau - 1), alpha <= C * tau, cp.sum(alpha) == 0]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)

    return alpha.value


# ==========================================
# 4. 归一化、训练与预测辅助函数
# ==========================================
# 固定归一化特征：只使用最初训练段，确保快速验证与测试阶段的尺度定义一致
train_feat_ref = Features_Raw[:lntr]
feat_min = np.min(train_feat_ref, axis=0)
feat_max = np.max(train_feat_ref, axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0


def normalize(X):
    return (X - feat_min) / feat_range


def fit_kernel_quantile_model(X_train_raw, Y_train, lambda_reg, sigma):
    X_train = normalize(X_train_raw)
    m = len(Y_train)
    C = 1.0 / (lambda_reg * m)
    gamma = 1.0 / (2 * sigma**2)

    K_train = rbf_kernel(X_train, gamma=gamma)
    alpha_val = solve_kernel_quantile_dual(K_train, Y_train, C, tau)

    if alpha_val is not None:
        support_indices = np.where((alpha_val > C * (tau - 1) + 1e-5) & (alpha_val < C * tau - 1e-5))[0]

        if len(support_indices) > 0:
            # 使用所有支持向量计算偏置并取平均，提高数值稳定性
            b_values = []
            for idx in support_indices:
                pred_no_b = np.dot(K_train[idx], alpha_val)
                b_values.append(Y_train[idx] - pred_no_b)
            b_val = float(np.mean(b_values))
        else:
            # 没有严格内部支持向量时，使用训练残差的中位数近似偏置
            b_val = float(np.median(Y_train - K_train @ alpha_val))
    else:
        # 求解失败 fallback
        alpha_val = np.zeros(m)
        b_val = float(np.quantile(Y_train, tau))

    return {"alpha": alpha_val, "b": b_val, "X_train": X_train, "gamma": gamma}


def predict_kernel_quantile(model, x_raw):
    x_norm = normalize(x_raw.reshape(1, -1))
    k_vector = rbf_kernel(model["X_train"], x_norm, gamma=model["gamma"]).flatten()
    q_pred = np.dot(model["alpha"], k_vector) + model["b"]
    return max(0.0, float(q_pred))


def evaluate_validation_once(lambda_reg, sigma):
    # 只在验证开始前训练一次固定模型，用它快速扫完整个验证段
    X_train_raw = Features_Raw[:lntr]
    Y_train = Demand[:lntr]
    model = fit_kernel_quantile_model(X_train_raw, Y_train, lambda_reg, sigma)

    val_features = Features_Raw[lntr : lntr + lnva]
    val_demand = Demand[lntr : lntr + lnva]

    q_pred = np.zeros(lnva)
    for i in range(lnva):
        q_pred[i] = predict_kernel_quantile(model, val_features[i])

    val_cost = nv_cost(q_pred, val_demand, b, h)
    return q_pred, val_cost


def run_test_rollout(start_idx, horizon, lambda_reg, sigma):
    Q_pred = np.zeros(horizon)
    Cost_realized = np.zeros(horizon)
    demand_eval = Demand[start_idx : start_idx + horizon]

    current_model = fit_kernel_quantile_model(
        Features_Raw[:lntr], Demand[:lntr], lambda_reg, sigma
    )
    for k in range(horizon):
        t = start_idx + k
        q_pred = predict_kernel_quantile(current_model, Features_Raw[t])
        Q_pred[k] = q_pred

        if t < TOTAL_LEN:
            actual_demand = Demand[t]
            Cost_realized[k] = nv_cost(q_pred, actual_demand, b, h)

    return Q_pred, Cost_realized, demand_eval


# ==========================================
# 5. 工程版快速选参 + 测试阶段滚动评估
# ==========================================
selection_records = []

print(
    "Start Kernel Quantile Regression "
    f"(lambda candidates={lambda_candidates}, sigma candidates={sigma_candidates}, update_step={update_step})..."
)
start_time = time.time()

for sigma in sigma_candidates:
    for lambda_reg in lambda_candidates:
        print(f"Validating static model: sigma={sigma}, lambda={lambda_reg}")
        _, val_cost = evaluate_validation_once(lambda_reg, sigma)
        val_mean_cost = float(np.mean(val_cost))
        selection_records.append({"sigma": sigma, "lambda_reg": lambda_reg, "validation_mean_cost": val_mean_cost})
        print(f"Validation mean cost (sigma={sigma}, lambda={lambda_reg}): {val_mean_cost:.6f}")

best_record = min(selection_records, key=lambda x: x["validation_mean_cost"])
best_sigma = best_record["sigma"]
best_lambda_reg = best_record["lambda_reg"]
best_val_cost = best_record["validation_mean_cost"]

print(
    f"Selected best hyperparameters: sigma={best_sigma}, lambda={best_lambda_reg} "
    f"(validation mean cost = {best_val_cost:.6f})"
)

start_idx = lntr + lnva
test_start_time = time.time()
Q_pred, Cost_realized, Demand_eval = run_test_rollout(start_idx, lnte, best_lambda_reg, best_sigma)
test_runtime_sec = time.time() - test_start_time
test_mean_cost = float(np.mean(Cost_realized))
print(f"Test mean cost with selected hyperparameters: {test_mean_cost:.6f}")
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
print(f"Test runtime: {test_runtime_sec:.2f} seconds")
record_runtime("RKHS", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 6. 保存结果 (CSV)
# ==========================================
output_filename = scenario_csv_path("nv_RKHS.csv", config)
selection_filename = scenario_csv_path("nv_RKHS_hyperparam_selection.csv", config)

df_out = pd.DataFrame(
    {
        "Selected_Sigma": np.full(lnte, best_sigma),
        "Selected_Lambda": np.full(lnte, best_lambda_reg),
        "Decision_Q": Q_pred,
        "Demand_D": Demand_eval,
        "Operation_Cost": Cost_realized,
    }
)

df_out.to_csv(output_filename, index=False)
df_selection = pd.DataFrame(selection_records)
df_selection["is_best"] = (df_selection["sigma"] == best_sigma) & (df_selection["lambda_reg"] == best_lambda_reg)
df_selection.to_csv(selection_filename, index=False)
print(f"Results saved to {output_filename}")
print(f"Hyperparameter selection summary saved to {selection_filename}")
