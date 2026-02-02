# ============================================================
# quantum_circuit_classifier.py
# ============================================================

from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional

import numpy as np
from scipy.sparse import csr_matrix

import os
import h5py

from Generate_quantum_circuit import (
    QuantumCircuitGenerator,
    CircuitGenConfig,
)

# ============================================================
# 等价类：数学上完全相同的量子线路
# ============================================================

@dataclass
class CircuitEquivalenceClass:
    representative_matrix: csr_matrix
    seeds: List[int] = field(default_factory=list)

    def add_seed(self, seed: int) -> None:
        self.seeds.append(seed)


# ============================================================
# 数据集：保存所有等价类 + 基本参数 + 非法 seed
# ============================================================

@dataclass
class CircuitDataset:
    L: int
    depth: int
    classes: List[CircuitEquivalenceClass] = field(default_factory=list)
    invalid_seeds: List[int] = field(default_factory=list)

    def num_classes(self) -> int:
        return len(self.classes)

    def total_valid_circuits(self) -> int:
        return sum(len(cls.seeds) for cls in self.classes)

    def total_invalid_circuits(self) -> int:
        return len(self.invalid_seeds)


# ============================================================
# 工具函数：稀疏矩阵严格相等判断
# ============================================================

def sparse_matrices_equal(A: csr_matrix, B: csr_matrix) -> bool:
    if A.shape != B.shape:
        return False
    diff = A != B
    return diff.nnz == 0


# ============================================================
# 主类：量子线路分类器
# ============================================================

class QuantumCircuitClassifier:
    def __init__(self, L: int, depth: int, dtype=np.complex128):
        self.config = CircuitGenConfig(L=L, depth=depth)
        self.generator = QuantumCircuitGenerator(self.config, dtype=dtype)
        self.dataset = CircuitDataset(L=L, depth=depth)

    # --------------------------------------------------------
    # 分类单个 seed
    # --------------------------------------------------------
    def classify_seed(self, seed: int) -> Optional[int]:
        try:
            _, H = self.generator.hamiltonian_from_seed(seed)
        except ValueError:
            self.dataset.invalid_seeds.append(seed)
            return None

        mat = csr_matrix(H.tocsc())

        for idx, eq_class in enumerate(self.dataset.classes):
            if sparse_matrices_equal(mat, eq_class.representative_matrix):
                eq_class.add_seed(seed)
                return idx

        new_class = CircuitEquivalenceClass(
            representative_matrix=mat,
            seeds=[seed],
        )
        self.dataset.classes.append(new_class)
        return len(self.dataset.classes) - 1

    # --------------------------------------------------------
    # 批量分类
    # --------------------------------------------------------
    def classify_seeds(self, seeds: Iterable[int], verbose: bool = False) -> Dict[int, Optional[int]]:
        result = {}
        for seed in seeds:
            cid = self.classify_seed(seed)
            result[seed] = cid
            if verbose:
                if cid is None:
                    print(f"seed {seed} -> 非法线路")
                else:
                    print(f"seed {seed} -> 等价类 {cid}")
        return result

    # --------------------------------------------------------
    # 常见用户操作
    # --------------------------------------------------------
    def print_dataset_summary(self) -> None:
        print("=== Quantum Circuit Dataset Summary ===")
        print(f"L = {self.dataset.L}")
        print(f"depth = {self.dataset.depth}")
        print(f"合法等价类数量 = {self.dataset.num_classes()}")
        print(f"合法线路数量 = {self.dataset.total_valid_circuits()}")
        print(f"非法线路数量 = {self.dataset.total_invalid_circuits()}")

    def print_class_info(self, class_index: int) -> None:
        cls = self.dataset.classes[class_index]
        print(f"--- Class {class_index} ---")
        print(f"seeds: {cls.seeds}")
        print(f"matrix shape: {cls.representative_matrix.shape}")

    def find_class_by_seed(self, seed: int) -> Optional[int]:
        for idx, cls in enumerate(self.dataset.classes):
            if seed in cls.seeds:
                return idx
        return None

    # --------------------------------------------------------
    # HDF5 保存
    # --------------------------------------------------------
    def save_to_hdf5(self, filename: str = "circuit_dataset.h5") -> None:
        save_dir = r"D:\OneDrive\Research\code\Machine_Learning\Optimize_quantum_circuits\data"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        with h5py.File(filepath, "w") as f:
            f.attrs["L"] = self.dataset.L
            f.attrs["depth"] = self.dataset.depth

            f.create_dataset("invalid_seeds", data=np.array(self.dataset.invalid_seeds, dtype=np.int64))

            classes_group = f.create_group("classes")

            for idx, cls in enumerate(self.dataset.classes):
                grp = classes_group.create_group(f"class_{idx}")

                grp.create_dataset("seeds", data=np.array(cls.seeds, dtype=np.int64))

                mat = cls.representative_matrix.tocsr()
                grp.create_dataset("matrix_data", data=mat.data)
                grp.create_dataset("matrix_indices", data=mat.indices)
                grp.create_dataset("matrix_indptr", data=mat.indptr)
                grp.attrs["matrix_shape"] = mat.shape

        print(f"HDF5 数据成功保存到：{filepath}")

    # --------------------------------------------------------
    # HDF5 加载
    # --------------------------------------------------------
    def load_from_hdf5(self, filename: str = "circuit_dataset.h5") -> None:
        load_dir = r"D:\OneDrive\Research\code\Machine_Learning\Optimize_quantum_circuits\data"
        filepath = os.path.join(load_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到 HDF5 文件：{filepath}")

        with h5py.File(filepath, "r") as f:
            L = int(f.attrs["L"])
            depth = int(f.attrs["depth"])

            self.dataset = CircuitDataset(L=L, depth=depth)

            self.dataset.invalid_seeds = list(f["invalid_seeds"][:])

            classes_group = f["classes"]

            for class_name in classes_group:
                grp = classes_group[class_name]

                seeds = list(grp["seeds"][:])

                data = grp["matrix_data"][:]
                indices = grp["matrix_indices"][:]
                indptr = grp["matrix_indptr"][:]
                shape = tuple(grp.attrs["matrix_shape"])

                mat = csr_matrix((data, indices, indptr), shape=shape)

                self.dataset.classes.append(
                    CircuitEquivalenceClass(
                        representative_matrix=mat,
                        seeds=seeds,
                    )
                )

        print(f"HDF5 数据成功加载：{filepath}")
