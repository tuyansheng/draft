# ============================================================
# circuit_classifier.py
# ============================================================

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from typing import Iterable, List

from Generate_quantum_circuit import (
    QuantumCircuitGenerator,
    CircuitGenConfig,
)

# ------------------------------------------------------------
# Sparse matrix helpers
# ------------------------------------------------------------
def save_csr(group: h5py.Group, mat: csr_matrix):
    group.create_dataset("data", data=mat.data)
    group.create_dataset("indices", data=mat.indices)
    group.create_dataset("indptr", data=mat.indptr)
    group.create_dataset("shape", data=mat.shape)


def load_csr(group: h5py.Group) -> csr_matrix:
    return csr_matrix(
        (group["data"][:],
         group["indices"][:],
         group["indptr"][:]),
        shape=tuple(group["shape"][:])
    )


def csr_equal(a: csr_matrix, b: csr_matrix) -> bool:
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


# ------------------------------------------------------------
# Main classifier
# ------------------------------------------------------------
class QuantumCircuitClassifier:
    def __init__(self, h5_path: str, cfg: CircuitGenConfig):
        self.h5_path = h5_path
        self.cfg = cfg
        self.gen = QuantumCircuitGenerator(cfg)

        with h5py.File(self.h5_path, "a") as f:
            self._init_file(f)

    # ---------------- initialization ----------------
    def _init_file(self, f: h5py.File):
        if "meta" not in f:
            meta = f.create_group("meta")
            meta.create_dataset("L", data=self.cfg.L)
            meta.create_dataset("depth", data=self.cfg.depth)

        f.require_group("equivalence_classes")
        f.require_group("illegal")

        if "seeds" not in f["illegal"]:
            f["illegal"].create_dataset(
                "seeds", shape=(0,), maxshape=(None,), dtype=np.int64
            )

        if "seed_index" not in f:
            idx = f.create_group("seed_index")
            idx.create_dataset(
                "all_seeds", shape=(0,), maxshape=(None,), dtype=np.int64
            )

    # ---------------- utilities ----------------
    def _append_seed(self, dset: h5py.Dataset, seed: int):
        n = len(dset)
        dset.resize((n + 1,))
        dset[n] = seed

    def _seed_exists(self, f: h5py.File, seed: int) -> bool:
        return seed in f["seed_index/all_seeds"][:]

    # ---------------- core logic ----------------
    def process_seed(self, seed: int):
        with h5py.File(self.h5_path, "a") as f:
            if self._seed_exists(f, seed):
                return

            # try generate circuit / Hamiltonian
            try:
                _, H = self.gen.hamiltonian_from_seed(seed)
                H_csr = H.tocsr()
            except Exception:
                self._append_seed(f["illegal/seeds"], seed)
                self._append_seed(f["seed_index/all_seeds"], seed)
                return

            eq_root = f["equivalence_classes"]

            # search existing equivalence classes
            for cls_name in eq_root:
                cls = eq_root[cls_name]
                H_ref = load_csr(cls["matrix"])
                if csr_equal(H_ref, H_csr):
                    self._append_seed(cls["seeds"], seed)
                    self._append_seed(f["seed_index/all_seeds"], seed)
                    return

            # create new equivalence class
            cls_id = len(eq_root)
            cls = eq_root.create_group(f"class_{cls_id:06d}")

            mat_grp = cls.create_group("matrix")
            save_csr(mat_grp, H_csr)

            cls.create_dataset(
                "seeds", data=np.array([seed], dtype=np.int64),
                maxshape=(None,)
            )

            self._append_seed(f["seed_index/all_seeds"], seed)

    # ---------------- batch operations ----------------
    def process_seeds(self, seeds: Iterable[int]):
        for s in seeds:
            self.process_seed(s)

    def process_range(self, start: int, stop: int):
        for s in range(start, stop):
            self.process_seed(s)

    # ============================================================
    # New: print all equivalence classes
    # ============================================================
    def print_equivalence_classes(self):
        with h5py.File(self.h5_path, "r") as f:
            eq_root = f["equivalence_classes"]
            print("\n=== Equivalence Classes ===")
            for cls_name in eq_root:
                seeds = list(eq_root[cls_name]["seeds"][:])
                print(f"{cls_name}: {seeds}")
            print("===========================\n")

    # ============================================================
    # New: print illegal seeds
    # ============================================================
    def print_illegal_seeds(self):
        with h5py.File(self.h5_path, "r") as f:
            illegal = list(f["illegal/seeds"][:])
            print("\n=== Illegal Seeds ===")
            print(illegal)
            print("=====================\n")

    # ============================================================
    # New: print full HDF5 structure
    # ============================================================
    def print_hdf5_structure(self):
        def _recurse(name, obj):
            indent = "  " * (name.count("/") - 1)
            if isinstance(obj, h5py.Group):
                print(f"{indent}[Group] {name}")
            else:
                print(f"{indent}[Dataset] {name} shape={obj.shape}")

        with h5py.File(self.h5_path, "r") as f:
            print("\n=== HDF5 Structure ===")
            f.visititems(_recurse)
            print("======================\n")

    # ============================================================
    # New: print details of a specific equivalence class
    # ============================================================
    def print_class_detail(self, class_name: str):
        with h5py.File(self.h5_path, "r") as f:
            if class_name not in f["equivalence_classes"]:
                print(f"[Error] Class {class_name} not found.")
                return

            cls = f["equivalence_classes"][class_name]
            seeds = list(cls["seeds"][:])

            print(f"\n=== Detail of {class_name} ===")
            print("Seeds:", seeds)

            # matrix info
            mat = load_csr(cls["matrix"])
            print("Matrix shape:", mat.shape)
            print("Matrix nnz:", mat.nnz)
            print("=============================\n")

    # ============================================================
    # New: find which class a seed belongs to
    # ============================================================
    def find_seed_class(self, seed: int):
        with h5py.File(self.h5_path, "r") as f:
            # check illegal
            if seed in f["illegal/seeds"][:]:
                print(f"Seed {seed} is ILLEGAL.")
                return "illegal"

            # search equivalence classes
            for cls_name in f["equivalence_classes"]:
                seeds = f["equivalence_classes"][cls_name]["seeds"][:]
                if seed in seeds:
                    print(f"Seed {seed} belongs to {cls_name}.")
                    return cls_name

            print(f"Seed {seed} not found.")
            return None

    # ============================================================
    # New: print circuit & Hamiltonian for a seed
    # ============================================================
    def print_seed_detail(self, seed: int):
        print(f"\n=== Seed {seed} Detail ===")
        try:
            circuit, H = self.gen.hamiltonian_from_seed(seed)
        except Exception as e:
            print("This seed is ILLEGAL:", e)
            return

        # print circuit
        print("\nCircuit:")
        self.gen.pretty_print(circuit)

        # print Hamiltonian info
        H_csr = H.tocsr()
        print("\nHamiltonian:")
        print("Shape:", H_csr.shape)
        print("Non-zero entries:", H_csr.nnz)
        print("==========================\n")
