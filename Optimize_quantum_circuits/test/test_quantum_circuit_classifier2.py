# ============================================================
# demo_equivalence_test.py
# ============================================================

import os
import h5py
from pprint import pprint

from circuit_classifier import QuantumCircuitClassifier
from Generate_quantum_circuit import CircuitGenConfig


# ------------------------------------------------------------
# Helper: print current equivalence classes
# ------------------------------------------------------------
def print_equivalence_classes(h5_path):
    with h5py.File(h5_path, "r") as f:
        print("\n=== Current Equivalence Classes ===")
        eq_root = f["equivalence_classes"]
        for cls_name in eq_root:
            seeds = list(eq_root[cls_name]["seeds"][:])
            print(f"{cls_name}: {seeds}")

        illegal = list(f["illegal/seeds"][:])
        print("\n=== Illegal Seeds ===")
        print(illegal)
        print("====================================\n")


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main():
    # 1) 配置量子线路
    cfg = CircuitGenConfig(L=2, depth=2)

    # 2) HDF5 文件
    h5_path = "demo_equiv_test.h5"
    if os.path.exists(h5_path):
        os.remove(h5_path)

    clf = QuantumCircuitClassifier(h5_path, cfg)

    # --------------------------------------------------------
    # Step A: 先处理一批 seeds，建立初始等价类
    # --------------------------------------------------------
    initial_seeds = [0, 1, 2, 3, 4, 5, 6]
    clf.process_seeds(initial_seeds)

    print(">>> 初始等价类：")
    print_equivalence_classes(h5_path)

    # --------------------------------------------------------
    # Step B: 选择特殊 seeds
    # --------------------------------------------------------

    # 1) 一个非法 seed（例如 CNOT 放在最后一行）
    illegal_seed = 5 ** (cfg.L * cfg.depth) - 1  # 最大 seed 通常非法概率高

    # 2) 一个等价类中已有的 seed（让它变成多元素类）
    #    例如 seed=0 的 Hamiltonian=0，很多 seed 也会产生 H=0
    duplicate_seed = 25  # 你可以随便挑，只要它合法

    # 3) 一个新的合法 seed，产生新的等价类
    new_class_seed = 7

    special_seeds = [illegal_seed, duplicate_seed, new_class_seed]

    print(">>> 特殊 seeds =", special_seeds)

    # --------------------------------------------------------
    # Step C: 加入特殊 seeds 前的状态
    # --------------------------------------------------------
    print(">>> 加入特殊 seeds 之前：")
    print_equivalence_classes(h5_path)

    # --------------------------------------------------------
    # Step D: 处理特殊 seeds
    # --------------------------------------------------------
    clf.process_seeds(special_seeds)

    # --------------------------------------------------------
    # Step E: 加入特殊 seeds 后的状态
    # --------------------------------------------------------
    print(">>> 加入特殊 seeds 之后：")
    print_equivalence_classes(h5_path)


if __name__ == "__main__":
    main()
