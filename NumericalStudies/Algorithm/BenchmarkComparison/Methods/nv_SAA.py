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

lntr = config["lntr"]
lnva = config["lnva"]
lnte = config["lnte"]
function_scenario = config["scenario"]
TOTAL_LEN = len(Demand)

print(f"已加载数据: {data_file} (Total: {TOTAL_LEN})")

# ==========================================
# 2. 参数设置
# ==========================================
b = 3 / 4
h = 1 / 4
r = b / (b + h)

# 结果存储
Q_pred = np.zeros(lnte)
TestSAA = np.zeros(lnte)


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环逻辑 (SAA)
# ==========================================
print(f"Start SAA Loop (Target Quantile: {r:.4f})")
start_time = time.time()

start_idx = lntr + lnva  # test beginning index

for k in range(lnte):
    t = start_idx + k

    # 获取训练数据 (滚动窗口)
    demand_train = Demand[:lntr]

    # SAA 求解
    q0 = np.quantile(demand_train, r)

    # 记录
    Q_pred[k] = q0

    # 样本外测试
    if t < TOTAL_LEN:
        actual_demand = Demand[t]
        TestSAA[k] = nv_cost(q0, actual_demand, b, h)

end_time = time.time()
test_runtime_sec = end_time - start_time
print(f"Loop finished in {test_runtime_sec:.4f} seconds")
record_runtime("SAA", test_runtime_sec, lnte, config_file=config_file)

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = scenario_csv_path("nv_SAA.csv", config)

# 构建 DataFrame
df_out = pd.DataFrame(
    {
        "Decision_Q": Q_pred,
        "Demand_D": Demand[start_idx : start_idx + lnte],
        "Operation_Cost": TestSAA,
    }
)

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")
