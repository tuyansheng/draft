# ============================================================
# test_seed_0.py
# ============================================================

import numpy as np
from Generate_quantum_circuit import CircuitGenConfig, QuantumCircuitGenerator


def main():
    # 1) 配置：你可以改 L 和 depth 再多测几种
    cfg = CircuitGenConfig(L=3, depth=3)
    gen = QuantumCircuitGenerator(cfg, dtype=np.complex128)

    seed = 0
    print(f"=== Test seed = {seed} ===")
    print(f"L = {cfg.L}, depth = {cfg.depth}")
    print(f"max_seed = {gen.max_seed}")
    print()

    # 2) 生成线路
    circuit = gen.circuit_from_seed(seed)

    print(">>> Circuit (pretty print):")
    gen.pretty_print(circuit)
    print()

    # 3) 逐元素打印线路矩阵（q, t, gate）
    print(">>> Circuit raw data (q, t, gate):")
    for q in range(cfg.L):
        for t in range(cfg.depth):
            print(f"q={q}, t={t}, gate={circuit[q][t]}")
    print()

    # 4) 生成 Hamiltonian
    circuit2, H = gen.hamiltonian_from_seed(seed)
    assert circuit2 == circuit  # sanity check

    print(">>> Hamiltonian basic info:")
    print("type:", type(H))
    print("basis size:", H.Ns)
    print("matrix shape:", H.shape)
    H_csr = H.tocsr()
    print("nnz (number of non-zero entries):", H_csr.nnz)
    print()

    # 5) 打印 Hamiltonian 稀疏矩阵的 data / indices / indptr（前若干项）
    print(">>> Hamiltonian CSR internal structure (first few entries):")
    print("data   :", H_csr.data[:10])
    print("indices:", H_csr.indices[:10])
    print("indptr :", H_csr.indptr[:10])
    print()

    # 6) 转成稠密矩阵看一眼（小 L 时可以，大 L 就别这么干了）
    if cfg.L <= 3:
        H_dense = H_csr.toarray()
        print(">>> Hamiltonian dense matrix:")
        print(H_dense)
        print()

        # 7) 看一下谱（特征值）
        evals = np.linalg.eigvalsh(H_dense)
        print(">>> Eigenvalues of H:")
        print(evals)
        print()

    print("=== Done. ===")


if __name__ == "__main__":
    main()
