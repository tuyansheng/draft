# =========================
# Generate_quantum_circuit.py
# =========================

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
        if seed !=0:
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
        elif seed ==0:
            circuit = self.circuit_from_seed(seed)
            static = [["I",[[1,0]]]]
            H = hamiltonian(static, [], basis=self.basis, dtype=self.dtype)
            return circuit, H

    # ---------- pretty print ----------
    def pretty_print(self, circuit: Circuit2D) -> None:
        T = len(circuit[0])
        print("     " + "".join(f"{t:>4}" for t in range(T)))
        print("     " + "----" * T)
        for q, row in enumerate(circuit):
            print(f"q{q:02d} " + "".join(f"{g:>4}" for g in row))
