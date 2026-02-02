# =========================
# test_generate_quantum_circuit.py
# =========================

import numpy as np

from Generate_quantum_circuit import (
    CircuitGenConfig,
    QuantumCircuitGenerator,
)


def test_basic_init():
    print("【测试1】初始化 QuantumCircuitGenerator 对象")
    cfg = CircuitGenConfig(L=3, depth=2)
    gen = QuantumCircuitGenerator(cfg, dtype=np.complex128)
    print(f"  - L = {gen.cfg.L}, depth = {gen.cfg.depth}")
    print(f"  - dtype = {gen.dtype}")
    print(f"  - basis 维度 = {gen.basis.Ns}")
    print(f"  - gate_map = {gen.gate_map}")
    print(f"  - max_seed = {gen.max_seed}")
    print("【结果】初始化成功\n")


def test_circuit_from_seed_valid():
    print("【测试2】circuit_from_seed：生成合法线路并打印")
    cfg = CircuitGenConfig(L=3, depth=2)
    gen = QuantumCircuitGenerator(cfg)

    seed = 10
    print(f"  - 使用 seed = {seed}")
    circuit = gen.circuit_from_seed(seed)
    print("  - 生成的线路（内部结构）：")
    for q, row in enumerate(circuit):
        print(f"    q{q}: {row}")

    print("  - 使用 pretty_print 打印线路：")
    gen.pretty_print(circuit)
    print("【结果】circuit_from_seed 正常工作\n")


def test_circuit_from_seed_range_check():
    print("【测试3】circuit_from_seed：seed 越界检查")
    cfg = CircuitGenConfig(L=2, depth=2)
    gen = QuantumCircuitGenerator(cfg)

    invalid_seeds = [-1, gen.max_seed]
    for s in invalid_seeds:
        print(f"  - 测试 seed = {s} 是否抛出异常")
        try:
            _ = gen.circuit_from_seed(s)
        except ValueError as e:
            print(f"    ✅ 正确抛出 ValueError：{e}")
        else:
            print("    ❌ 未抛出异常（错误）")
    print("【结果】seed 越界检查逻辑正常\n")


def test_validate_circuit_cnot_rules():
    print("【测试4】_validate_circuit：CNOT 规则检查")

    cfg = CircuitGenConfig(L=3, depth=3)
    gen = QuantumCircuitGenerator(cfg)

    # 合法线路：CNOT 下一行是 I，最后一行没有 CNOT
    valid_circuit = [
        ["CNOT", "I", "Z"],  # q0
        ["I",    "X", "I"],  # q1
        ["Z",    "Y", "X"],  # q2 (最后一行不能有 CNOT)
    ]
    print("  - 测试合法线路：应通过验证")
    try:
        gen._validate_circuit(valid_circuit)
        print("    ✅ 合法线路验证通过")
    except ValueError as e:
        print(f"    ❌ 不应抛出异常，但得到：{e}")

    # 非法线路1：CNOT 下一行不是 I
    invalid_circuit_1 = [
        ["CNOT", "I", "Z"],  # q0
        ["X",    "X", "I"],  # q1 (这里应为 I 才合法)
        ["Z",    "Y", "X"],
    ]
    print("  - 测试非法线路1：CNOT 下一行不是 I，应抛出异常")
    try:
        gen._validate_circuit(invalid_circuit_1)
    except ValueError as e:
        print(f"    ✅ 正确抛出异常：{e}")
    else:
        print("    ❌ 未抛出异常（错误）")

    # 非法线路2：最后一行有 CNOT
    invalid_circuit_2 = [
        ["I",    "I",    "Z"],
        ["I",    "CNOT", "I"],
        ["Z",    "I",    "CNOT"],  # 最后一行出现 CNOT
    ]
    print("  - 测试非法线路2：最后一行有 CNOT，应抛出异常")
    try:
        gen._validate_circuit(invalid_circuit_2)
    except ValueError as e:
        print(f"    ✅ 正确抛出异常：{e}")
    else:
        print("    ❌ 未抛出异常（错误）")

    print("【结果】CNOT 规则验证逻辑正常\n")


def test_hamiltonian_from_seed():
    print("【测试5】hamiltonian_from_seed：生成 Hamiltonian 并检查基本性质")
    cfg = CircuitGenConfig(L=3, depth=2)
    gen = QuantumCircuitGenerator(cfg)

    seed = 7
    print(f"  - 使用 seed = {seed}")
    circuit, H = gen.hamiltonian_from_seed(seed)

    print("  - 生成的线路：")
    gen.pretty_print(circuit)

    print("  - Hamiltonian 基本信息：")
    print(f"    类型：{type(H)}")
    print(f"    维度：{H.Ns} x {H.Ns}")
    print(f"    dtype：{H.dtype}")

    # 简单数值检查：取一个随机态，计算 <psi|H|psi>
    psi = np.random.randn(H.Ns) + 1j * np.random.randn(H.Ns)
    psi = psi / np.linalg.norm(psi)
    exp_val = np.vdot(psi, H.dot(psi))
    print(f"  - 随机态期望值 <psi|H|psi> = {exp_val}")

    print("【结果】hamiltonian_from_seed 正常工作\n")


def test_dtype_check():
    print("【测试6】dtype 检查：包含 Y 门时必须使用复数 dtype")
    cfg = CircuitGenConfig(L=2, depth=1)

    print("  - 使用复数 dtype=np.complex128：应当通过")
    try:
        _ = QuantumCircuitGenerator(cfg, dtype=np.complex128)
        print("    ✅ 正常初始化")
    except ValueError as e:
        print(f"    ❌ 不应抛出异常，但得到：{e}")

    print("  - 使用实数 dtype=float：应当抛出异常")
    try:
        _ = QuantumCircuitGenerator(cfg, dtype=float)
    except ValueError as e:
        print(f"    ✅ 正确抛出异常：{e}")
    else:
        print("    ❌ 未抛出异常（错误）")

    print("【结果】dtype 检查逻辑正常\n")


if __name__ == "__main__":
    print("========== 开始运行 QuantumCircuitGenerator 全面测试 ==========\n")
    test_basic_init()
    test_circuit_from_seed_valid()
    test_circuit_from_seed_range_check()
    test_validate_circuit_cnot_rules()
    test_hamiltonian_from_seed()
    test_dtype_check()
    print("========== 所有测试执行完毕，请检查上方输出是否符合预期 ==========")
