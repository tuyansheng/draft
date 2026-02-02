# =========================
# Generate_quantum_circuit.py
# =========================

import os
import pickle
import random
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

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
    p_identity: float = 0.4
    p_single_pauli: float = 0.4
    p_cnot: float = 0.2


# -------------------------
# Generator: seed -> circuit -> Hamiltonian
# -------------------------
class QuantumCircuitGenerator:
    def __init__(self, cfg: CircuitGenConfig, dtype=np.complex128):
        self.cfg = cfg
        self.dtype = dtype
        self.basis = spin_basis_1d(cfg.L)

        if dtype not in (np.complex64, np.complex128, complex):
            raise ValueError("包含 Y 门的线路必须使用复数 dtype")

    # ---------- validation ----------
    @staticmethod
    def _validate_circuit(circuit: Circuit2D) -> Tuple[int, int]:
        if not circuit or not all(isinstance(r, list) for r in circuit):
            raise ValueError("circuit 必须是二维 list")

        L = len(circuit)
        T = len(circuit[0])
        if T <= 0:
            raise ValueError("depth 必须 > 0")
        if any(len(r) != T for r in circuit):
            raise ValueError("circuit 每一行长度必须一致")

        allowed = {"I", "X", "Y", "Z", "CNOT"}
        for q in range(L):
            for t in range(T):
                if circuit[q][t] not in allowed:
                    raise ValueError(f"非法门 {circuit[q][t]}")

        for q in range(L - 1):
            for t in range(T):
                if circuit[q][t] == "CNOT" and circuit[q + 1][t] != "I":
                    raise ValueError("CNOT 下一行必须是 I")

        if any(circuit[L - 1][t] == "CNOT" for t in range(T)):
            raise ValueError("最后一行不能放 CNOT")

        return L, T

    # ---------- seed -> circuit ----------
    def circuit_from_seed(self, seed: int) -> Circuit2D:
        rng = random.Random(seed)
        L, T = self.cfg.L, self.cfg.depth
        circuit: Circuit2D = [["I"] * T for _ in range(L)]

        for t in range(T):
            q = 0
            while q < L:
                if q == L - 1:
                    circuit[q][t] = rng.choice(["I", "X", "Y", "Z"])
                    q += 1
                    continue

                r = rng.random()
                if r < self.cfg.p_identity:
                    g = "I"
                elif r < self.cfg.p_identity + self.cfg.p_single_pauli:
                    g = rng.choice(["X", "Y", "Z"])
                else:
                    g = "CNOT"

                if g == "CNOT":
                    circuit[q][t] = "CNOT"
                    circuit[q + 1][t] = "I"
                    q += 2
                else:
                    circuit[q][t] = g
                    q += 1

        self._validate_circuit(circuit)
        return circuit

    # ---------- seed -> Hamiltonian ----------
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
    def pretty_print(self, circuit: Circuit2D):
        L, T = self._validate_circuit(circuit)
        print("     " + "".join(f"{t:>4}" for t in range(T)))
        print("     " + "----" * T)
        for q in range(L):
            print(f"q{q:02d} " + "".join(f"{circuit[q][t]:>4}" for t in range(T)))


# -------------------------
# DB: one DB per (L, depth)
# -------------------------
class QuSpinCircuitDB:
    def __init__(self, cfg: CircuitGenConfig):
        self.cfg = cfg

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = os.path.join(
            data_dir, f"circuits_L{cfg.L}_T{cfg.depth}.db"
        )
        self._init_db()

    def _init_db(self):
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
# Batch generation
# -------------------------
def generate_and_store_batch(
    gen: QuantumCircuitGenerator,
    db: QuSpinCircuitDB,
    n: int,
    seed_start: int = 0,
) -> int:
    inserted = 0
    seed = seed_start

    while inserted < n:
        if db.has_seed(seed):
            seed += 1
            continue

        _, H = gen.hamiltonian_from_seed(seed)
        if db.insert(seed, H):
            inserted += 1

        seed += 1

    return inserted


# -------------------------
# Main: explicit checks
# -------------------------
if __name__ == "__main__":
    cfg = CircuitGenConfig(L=5, depth=8)
    gen = QuantumCircuitGenerator(cfg)
    db = QuSpinCircuitDB(cfg)

    # 1. Determinism check
    c1 = gen.circuit_from_seed(42)
    c2 = gen.circuit_from_seed(42)
    assert c1 == c2, "同 seed 生成的线路不一致"

    # 2. Hamiltonian construction check
    _, H = gen.hamiltonian_from_seed(42)
    assert H.Ns == spin_basis_1d(cfg.L).Ns, "Hilbert 空间维数错误"

    # 3. Database insertion
    inserted = generate_and_store_batch(gen, db, n=3)
    print("Inserted:", inserted)

    # 4. Load & inspect
    seed_demo = 0
    circuit = gen.circuit_from_seed(seed_demo)
    gen.pretty_print(circuit)

    H_loaded = db.load(seed_demo)
    print("Hamiltonian type:", type(H_loaded))
    print("Hilbert dimension:", H_loaded.Ns)
