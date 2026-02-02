# =========================
# Generate_quantum_circuit.py
# =========================

import os
import json
import pickle
import random
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


Gate = str
Circuit2D = List[List[Gate]]


def _validate_circuit(circuit: Circuit2D) -> Tuple[int, int]:
    if not circuit or not isinstance(circuit, list) or not all(isinstance(r, list) for r in circuit):
        raise ValueError("circuit 必须是二维 list。")

    L = len(circuit)
    T = len(circuit[0])
    if T == 0:
        raise ValueError("circuit 的列数必须 > 0。")
    if any(len(r) != T for r in circuit):
        raise ValueError("circuit 每一行长度必须一致。")

    allowed = {"I", "X", "Y", "Z", "CNOT"}
    for q in range(L):
        for t in range(T):
            g = circuit[q][t]
            if g not in allowed:
                raise ValueError(f"非法门 {g}")

    for q in range(L - 1):
        for t in range(T):
            if circuit[q][t] == "CNOT" and circuit[q + 1][t] != "I":
                raise ValueError("CNOT 下一行必须是 I")

    if any(circuit[L - 1][t] == "CNOT" for t in range(T)):
        raise ValueError("最后一行不能放 CNOT")

    return L, T


def circuit_to_quspin_hamiltonian(
    circuit: Circuit2D,
    basis=None,
    dtype=np.float64,
):
    L, T = _validate_circuit(circuit)
    if basis is None:
        basis = spin_basis_1d(L)

    static = []

    for q in range(L):
        for t in range(T):
            g = circuit[q][t]
            if g == "X":
                static.append(["x", [[1.0, q]]])
            elif g == "Y":
                static.append(["y", [[1.0, q]]])
            elif g == "Z":
                static.append(["z", [[1.0, q]]])

    c = np.pi / 4.0
    for q in range(L - 1):
        for t in range(T):
            if circuit[q][t] == "CNOT":
                static.append(["z", [[-c, q]]])
                static.append(["x", [[-c, q + 1]]])
                static.append(["zx", [[+c, q, q + 1]]])

    return hamiltonian(static, [], basis=basis)


@dataclass
class CircuitGenConfig:
    depth: int
    p_cnot: float = 0.2
    p_identity: float = 0.4
    p_single_pauli: float = 0.4


class QuSpinCircuitDB:
    def __init__(self, L: int, db_path: str, dtype=np.float64):
        self.L = L
        self.db_path = db_path
        self.dtype = dtype
        self.basis = spin_basis_1d(L)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS circuits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    circuit_json TEXT UNIQUE,
                    hamiltonian_blob BLOB
                );
                """
            )
            con.commit()

    @staticmethod
    def _canonical_json(circuit: Circuit2D) -> str:
        return json.dumps(circuit, separators=(",", ":"))

    def _exists(self, cj: str) -> bool:
        with sqlite3.connect(self.db_path) as con:
            return con.execute(
                "SELECT 1 FROM circuits WHERE circuit_json=? LIMIT 1", (cj,)
            ).fetchone() is not None

    def generate_unique_circuits(
        self,
        n: int,
        cfg: CircuitGenConfig,
        seed: Optional[int] = None,
    ) -> List[Circuit2D]:
        rng = random.Random(seed)
        seen = set()
        batch = []

        while len(batch) < n:
            circuit = [["I"] * cfg.depth for _ in range(self.L)]

            for t in range(cfg.depth):
                q = 0
                while q < self.L:
                    if q == self.L - 1:
                        circuit[q][t] = rng.choice(["I", "X", "Y", "Z"])
                        q += 1
                        continue

                    r = rng.random()
                    if r < cfg.p_identity:
                        g = "I"
                    elif r < cfg.p_identity + cfg.p_single_pauli:
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

            _validate_circuit(circuit)
            cj = self._canonical_json(circuit)

            if cj not in seen and not self._exists(cj):
                seen.add(cj)
                batch.append(circuit)

        return batch

    def store_circuits_with_hamiltonians(self, circuits: List[Circuit2D]) -> int:
        inserted = 0
        with sqlite3.connect(self.db_path) as con:
            for circuit in circuits:
                cj = self._canonical_json(circuit)
                if self._exists(cj):
                    continue

                H = circuit_to_quspin_hamiltonian(
                    circuit, basis=self.basis, dtype=self.dtype
                )
                blob = pickle.dumps(H, protocol=pickle.HIGHEST_PROTOCOL)

                cur = con.execute(
                    "INSERT OR IGNORE INTO circuits VALUES(NULL, ?, ?)",
                    (cj, sqlite3.Binary(blob)),
                )
                if cur.rowcount == 1:
                    inserted += 1
            con.commit()
        return inserted

    def pretty_print(self, circuit: Circuit2D):
        L, T = _validate_circuit(circuit)
        print("     " + "".join(f"{t:>4}" for t in range(T)))
        print("     " + "----" * T)
        for q in range(L):
            print(f"q{q:02d} " + "".join(f"{circuit[q][t]:>4}" for t in range(T)))

    def load_hamiltonian_by_circuit(self, circuit: Circuit2D):
        cj = self._canonical_json(circuit)
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT hamiltonian_blob FROM circuits WHERE circuit_json=?",
                (cj,),
            ).fetchone()
            return pickle.loads(row[0]) if row else None


if __name__ == "__main__":
    L = 5
    depth = 8
    db_path = r"D:\OneDrive\Research\code\Machine_Learning\Optimize_quantum_circuits\data\circuits.db"

    db = QuSpinCircuitDB(L, db_path)

    cfg = CircuitGenConfig(depth=depth)
    circuits = db.generate_unique_circuits(3, cfg, seed=42)

    db.pretty_print(circuits[0])

    inserted = db.store_circuits_with_hamiltonians(circuits)
    print("Inserted:", inserted)

    H = db.load_hamiltonian_by_circuit(circuits[0])
    print("Hamiltonian type:", type(H))
    print("Hilbert dimension:", H.Ns)
