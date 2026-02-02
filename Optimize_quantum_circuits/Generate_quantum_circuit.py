# =========================
# Generate_quantum_circuit.py
# =========================

import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import List

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


Gate = str
Circuit2D = List[List[Gate]]


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class CircuitGenConfig:
    L: int
    depth: int


# -------------------------
# Generator: seed → circuit → Hamiltonian (ENUMERATION)
# -------------------------
class QuantumCircuitGenerator:
    def __init__(self, cfg: CircuitGenConfig, dtype=np.complex128):
        if dtype not in (np.complex64, np.complex128, complex):
            raise ValueError("包含 Y 门的线路必须使用复数 dtype")

        self.cfg = cfg
        self.dtype = dtype
        self.basis = spin_basis_1d(cfg.L)

        self.gate_map = ["I", "X", "Y", "Z", "CNOT"]
        self.max_seed = 5 ** (cfg.L * cfg.depth)

    # ---------- validation ----------
    @staticmethod
    def _validate_circuit(circuit: Circuit2D) -> None:
        L = len(circuit)
        T = len(circuit[0])

        for q in range(L - 1):
            for t in range(T):
                if circuit[q][t] == "CNOT" and circuit[q + 1][t] != "I":
                    raise ValueError("非法线路：CNOT 下一行必须是 I")

        if any(circuit[L - 1][t] == "CNOT" for t in range(T)):
            raise ValueError("非法线路：最后一行不能放 CNOT")

    # ---------- seed → circuit (base-5 enumeration) ----------
    def circuit_from_seed(self, seed: int) -> Circuit2D:
        if seed < 0 or seed >= self.max_seed:
            raise ValueError(
                f"seed 超出范围，应满足 0 <= seed < {self.max_seed}"
            )

        L, T = self.cfg.L, self.cfg.depth

        # base-5 展开
        digits = []
        x = seed
        for _ in range(L * T):
            digits.append(x % 5)
            x //= 5

        # 构造线路（t 优先，再 q）
        circuit: Circuit2D = [["I"] * T for _ in range(L)]
        idx = 0
        for t in range(T):
            for q in range(L):
                circuit[q][t] = self.gate_map[digits[idx]]
                idx += 1

        self._validate_circuit(circuit)
        return circuit

    # ---------- seed → Hamiltonian ----------
    def hamiltonian_from_seed(self, seed: int):
        circuit = self.circuit_from_seed(seed)
        static = []

        for q in range(self.cfg.L):
            for t in range(self.cfg.depth):
                g = circuit[q][t]
                if g == "X":
                    static.append(["x", [[1.0, q]]])
                elif g == "Y":
                    static.append(["y", [[1.0, q]]])
                elif g == "Z":
                    static.append(["z", [[1.0, q]]])

        c = np.pi / 4.0
        for q in range(self.cfg.L - 1):
            for t in range(self.cfg.depth):
                if circuit[q][t] == "CNOT":
                    static.append(["z", [[-c, q]]])
                    static.append(["x", [[-c, q + 1]]])
                    static.append(["zx", [[+c, q, q + 1]]])

        H = hamiltonian(static, [], basis=self.basis, dtype=self.dtype)
        return circuit, H

    # ---------- pretty print ----------
    def pretty_print(self, circuit: Circuit2D) -> None:
        T = len(circuit[0])
        print("     " + "".join(f"{t:>4}" for t in range(T)))
        print("     " + "----" * T)
        for q, row in enumerate(circuit):
            print(f"q{q:02d} " + "".join(f"{g:>4}" for g in row))


# -------------------------
# DB: one DB per (L, depth)
# -------------------------
class QuSpinCircuitDB:
    def __init__(self, cfg: CircuitGenConfig, base_dir: str | None = None):
        self.cfg = cfg

        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = os.path.join(
            data_dir, f"circuits_L{cfg.L}_T{cfg.depth}.db"
        )
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS circuits (
                    seed INTEGER PRIMARY KEY,
                    hamiltonian_blob BLOB NOT NULL
                );
                """
            )
            con.commit()

    def has_seed(self, seed: int) -> bool:
        with sqlite3.connect(self.db_path) as con:
            return con.execute(
                "SELECT 1 FROM circuits WHERE seed=? LIMIT 1",
                (seed,),
            ).fetchone() is not None

    def insert(self, seed: int, H) -> bool:
        blob = pickle.dumps(H, protocol=pickle.HIGHEST_PROTOCOL)
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO circuits VALUES (?, ?)",
                (seed, sqlite3.Binary(blob)),
            )
            con.commit()
        return cur.rowcount == 1

    def load(self, seed: int):
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT hamiltonian_blob FROM circuits WHERE seed=?",
                (seed,),
            ).fetchone()
        return pickle.loads(row[0]) if row else None


# -------------------------
# Batch enumeration helper
# -------------------------
def generate_and_store_batch(
    gen: QuantumCircuitGenerator,
    db: QuSpinCircuitDB,
    n: int,
    seed_start: int = 0,
) -> int:
    inserted = 0
    seed = seed_start

    while inserted < n and seed < gen.max_seed:
        if db.has_seed(seed):
            seed += 1
            continue

        try:
            _, H = gen.hamiltonian_from_seed(seed)
        except ValueError:
            seed += 1
            continue  # 非法线路，跳过

        if db.insert(seed, H):
            inserted += 1

        seed += 1

    return inserted
