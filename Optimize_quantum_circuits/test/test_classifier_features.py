# ============================================================
# test_classifier_features.py
# ============================================================

import os
from circuit_classifier import QuantumCircuitClassifier
from Generate_quantum_circuit import CircuitGenConfig


def main():
    # -----------------------------
    # 1. 创建一个新的 HDF5 文件
    # -----------------------------
    h5_path = "D:\\OneDrive\\Research\\code\\Machine_Learning\\Optimize_quantum_circuits\\data\\datatest_classifier.h5"
    if os.path.exists(h5_path):
        os.remove(h5_path)

    cfg = CircuitGenConfig(L=3, depth=4)
    clf = QuantumCircuitClassifier(h5_path, cfg)

    # -----------------------------
    # 2. 处理一些 seeds
    # -----------------------------
    seeds_to_test = [0]

    # -----------------------------
    # 3. 测试新增功能
    # ----------------------------

    print("\n==============================")
    print("Test: print_seed_detail")
    print("==============================")
    clf.print_seed_detail(0)

    print("\n==============================")
    print("Test: print_seed_detail")
    print("==============================")
    clf.print_seed_detail(1)

if __name__ == "__main__":
    main()
