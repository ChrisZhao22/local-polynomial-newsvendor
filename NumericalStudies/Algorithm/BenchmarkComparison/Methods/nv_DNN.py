import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import json
import os
import time

import numpy as np
import pandas as pd
from feature_loader import load_feature_matrix
from runtime_logger import record_runtime
from simulation_paths import scenario_csv_path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

lntr = int(config["lntr"])
lnva = int(config["lnva"])
lnte = int(config["lnte"])
function_scenario = config["scenario"]
TOTAL_LEN = len(Demand)

print(f"已加载数据. Features shape: {Features_Raw.shape}, Total Samples: {TOTAL_LEN}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 3 / 4
h = 1 / 4


# 保持与Deep Neural Newsvendor推荐的参数设置一致

hidden_candidates = config.get("dnn_hidden_candidates", [[512, 512, 512]])
lr_candidates = [float(v) for v in config.get("dnn_lr_candidates", [1e-3,0.0005])]
weight_decay_candidates = [float(v) for v in config.get("dnn_weight_decay_candidates", [0.0,1e-5])]
update_step = int(config.get("dnn_update_step", 80))
batch_size = int(config.get("dnn_batch_size", 128))
max_epochs = int(config.get("dnn_max_epochs", 100))
patience = int(config.get("dnn_patience", 20))
min_delta = float(config.get("dnn_min_delta", 1e-4))
val_ratio = float(config.get("dnn_val_ratio", 0.2))
seed = int(config.get("dnn_seed", 42))
device_name = str(config.get("dnn_device", "cpu")).lower()

print(
    "DNN参数候选: "
    f"hidden={hidden_candidates}, lr={lr_candidates}, wd={weight_decay_candidates}, update_step={update_step}"
)

# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


def build_dnn_net(input_dim, hidden_layers):
    width_vec = [input_dim, *hidden_layers, 1]
    modules = []
    for i in range(len(width_vec) - 2):
        modules.append(nn.Linear(width_vec[i], width_vec[i + 1]))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(width_vec[-2], width_vec[-1]))
    return nn.Sequential(*modules)


def nv_loss_torch(y_true, y_pred, b, h):
    diff = y_true - y_pred
    idx = (diff > 0).float()
    loss = h * torch.abs(diff) * (1 - idx) + b * idx * torch.abs(diff)
    return loss.mean()


def standardize_train_eval(x_train, x_eval):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_eval - mean) / std


def fit_predict_dnn_block(x_window, y_window, x_eval_block, hidden_layers, lr, weight_decay, seed_value):
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(int(seed_value))

    n = len(y_window)
    val_size = int(round(n * val_ratio))
    val_size = min(max(val_size, 1), n - 1)
    split_idx = n - val_size

    x_train = x_window[:split_idx]
    y_train = y_window[:split_idx]
    x_val = x_window[split_idx:]
    y_val = y_window[split_idx:]

    x_train_s, x_val_s = standardize_train_eval(x_train, x_val)
    _, x_eval_s = standardize_train_eval(x_train, x_eval_block)

    train_dataset = TensorDataset(
        torch.tensor(x_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=min(max(1, int(batch_size)), len(train_dataset)),
        shuffle=True,
        num_workers=0,
    )

    x_val_t = torch.tensor(x_val_s, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
    x_eval_t = torch.tensor(x_eval_s, dtype=torch.float32, device=device)

    net = build_dnn_net(x_window.shape[1], hidden_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = float("inf")
    best_epoch = 1
    best_state = None
    bad_epochs = 0

    for epoch in range(int(max_epochs)):
        net.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred_train = net(x_batch)
            loss_train = nv_loss_torch(y_batch, pred_train, b=b, h=h)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            pred_val = net(x_val_t)
            val_loss = nv_loss_torch(y_val_t, pred_val, b=b, h=h)
            val_value = float(val_loss.item())

        if val_value < best_val - float(min_delta):
            best_val = val_value
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        pred_eval = net(x_eval_t).detach().cpu().numpy().reshape(-1)

    pred_eval = np.maximum(pred_eval, 0.0)
    return pred_eval, int(best_epoch), float(best_val)


def run_dnn_rollout(start_idx, horizon, hidden_layers, lr, weight_decay, show_progress=False):
    Q_pred = np.zeros(horizon)
    operation_cost = np.zeros(horizon)
    demand_eval = Demand[start_idx : start_idx + horizon]
    if show_progress:
        print(f"DNN fixed-split fit for {horizon} evaluation points")
    q_block, best_epoch, best_val = fit_predict_dnn_block(
        x_window=Features_Raw[:lntr],
        y_window=Demand[:lntr],
        x_eval_block=Features_Raw[start_idx : start_idx + horizon],
        hidden_layers=hidden_layers,
        lr=lr,
        weight_decay=weight_decay,
        seed_value=seed + start_idx,
    )
    Q_pred[:] = q_block
    operation_cost[:] = np.asarray(
        nv_cost(q_block, Demand[start_idx : start_idx + horizon], b, h), dtype=float
    )
    retrain_df = pd.DataFrame(
        [{
            "start_test_index": start_idx,
            "block_size": horizon,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }]
    )
    return Q_pred, operation_cost, demand_eval, retrain_df


# ==========================================
# 4. 先在验证集选超参数，再在测试集评估
# ==========================================
selection_records = []
overall_start_time = time.time()

candidate_grid = []
for hidden_layers in hidden_candidates:
    for lr in lr_candidates:
        for wd in weight_decay_candidates:
            candidate_grid.append((hidden_layers, lr, wd))

for idx, (hidden_layers, lr, wd) in enumerate(candidate_grid, start=1):
    hidden_str = "-".join(str(v) for v in hidden_layers)
    print(f"DNN validating [{idx}/{len(candidate_grid)}]: hidden={hidden_str}, lr={lr}, wd={wd}")
    _, val_cost, _, _ = run_dnn_rollout(
        start_idx=lntr,
        horizon=lnva,
        hidden_layers=hidden_layers,
        lr=lr,
        weight_decay=wd,
        show_progress=False,
    )
    val_mean_cost = float(np.mean(val_cost))
    selection_records.append(
        {
            "hidden_layers": hidden_str,
            "lr": lr,
            "weight_decay": wd,
            "validation_mean_cost": val_mean_cost,
        }
    )
    print(f"Validation mean cost (hidden={hidden_str}, lr={lr}, wd={wd}): {val_mean_cost:.6f}")

best_record = min(selection_records, key=lambda x: x["validation_mean_cost"])
best_hidden_str = best_record["hidden_layers"]
best_hidden = [int(v) for v in str(best_hidden_str).split("-")]
best_lr = float(best_record["lr"])
best_wd = float(best_record["weight_decay"])
best_val_cost = float(best_record["validation_mean_cost"])

print(
    f"Selected best hyperparameters: hidden={best_hidden_str}, lr={best_lr}, wd={best_wd} "
    f"(validation mean cost={best_val_cost:.6f})"
)

test_start_idx = lntr + lnva
test_start_time = time.time()
Q_pred, operation_cost, Demand_eval, retrain_df = run_dnn_rollout(
    start_idx=test_start_idx,
    horizon=lnte,
    hidden_layers=best_hidden,
    lr=best_lr,
    weight_decay=best_wd,
    show_progress=True,
)
test_runtime_sec = time.time() - test_start_time
test_mean_cost = float(np.mean(operation_cost))

print(f"Total time: {time.time() - overall_start_time:.2f} s")
print(f"Test runtime: {test_runtime_sec:.2f} s")
print(f"Test mean cost with selected hyperparameters: {test_mean_cost:.6f}")
record_runtime("DNN (Deep_NV)", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = scenario_csv_path("nv_DNN.csv", config)
selection_filename = scenario_csv_path("nv_DNN_hyperparam_selection.csv", config)
retrain_filename = scenario_csv_path("nv_DNN_retrain_summary.csv", config)

df_out = pd.DataFrame(
    {
        "Selected_Hidden_Layers": np.full(lnte, best_hidden_str),
        "Selected_LR": np.full(lnte, best_lr),
        "Selected_Weight_Decay": np.full(lnte, best_wd),
        "Selected_Update_Step": np.full(lnte, update_step),
        "Decision_Q": Q_pred,
        "Demand_D": Demand_eval,
        "Operation_Cost": operation_cost,
    }
)
df_out.to_csv(output_filename, index=False)

df_selection = pd.DataFrame(selection_records)
df_selection["is_best"] = (
    (df_selection["hidden_layers"] == best_hidden_str)
    & (df_selection["lr"] == best_lr)
    & (df_selection["weight_decay"] == best_wd)
)
df_selection.to_csv(selection_filename, index=False)
retrain_df.to_csv(retrain_filename, index=False)

print(f"Results saved to {output_filename}")
print(f"Hyperparameter selection summary saved to {selection_filename}")
print(f"Retraining summary saved to {retrain_filename}")
