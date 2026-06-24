import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from linear_model_benchmark import run_linear_benchmark


if __name__ == "__main__":
    run_linear_benchmark(penalty="l1", model_name="LinearModel-L1", output_stem="nv_LinearModel_L1")
