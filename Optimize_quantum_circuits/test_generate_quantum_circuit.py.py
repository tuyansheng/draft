# =========================
# test_generate_quantum_circuit.py
# =========================

print("开始测试 Generate_quantum_circuit 模块...\n")

# -------------------------
# 1. 导入模块
# -------------------------
print("【步骤 1】测试模块导入...")

from Generate_quantum_circuit import (
    CircuitGenConfig,
    QuantumCircuitGenerator,
    QuSpinCircuitDB,
    generate_and_store_batch,
)

print("模块导入成功。\n")


# -------------------------
# 2. 创建配置
# -------------------------
print("【步骤 2】创建线路生成配置...")

cfg = CircuitGenConfig(
    L=5,
    depth=8,
    p_identity=0.4,
    p_single_pauli=0.4,
    p_cnot=0.2,
)

print("配置创建成功：")
print(cfg, "\n")


# -------------------------
# 3. 创建线路生成器
# -------------------------
print("【步骤 3】创建 QuantumCircuitGenerator...")

gen = QuantumCircuitGenerator(cfg)

print("生成器创建成功。")
print(f"系统量子比特数 L = {cfg.L}")
print(f"Hilbert 空间维数 = {gen.basis.Ns}\n")


# -------------------------
# 4. 测试 seed → circuit 的确定性
# -------------------------
print("【步骤 4】测试同一个 seed 是否生成相同线路...")

seed_test = 123
circuit_1 = gen.circuit_from_seed(seed_test)
circuit_2 = gen.circuit_from_seed(seed_test)

if circuit_1 == circuit_2:
    print(f"seed = {seed_test} 的线路生成是确定性的 ✔")
else:
    print("错误：同一个 seed 生成了不同线路 ✘")

print("\n生成的线路如下：")
gen.pretty_print(circuit_1)
print()


# -------------------------
# 5. 测试 seed → Hamiltonian
# -------------------------
print("【步骤 5】测试 Hamiltonian 构造...")

_, H = gen.hamiltonian_from_seed(seed_test)

print("Hamiltonian 构造成功。")
print("Hamiltonian 类型：", type(H))
print("Hilbert 空间维数：", H.Ns)
print("是否为复数矩阵：", H.dtype)
print()


# -------------------------
# 6. 创建数据库
# -------------------------
print("【步骤 6】创建数据库对象...")

db = QuSpinCircuitDB(cfg)

print("数据库创建成功。")
print("数据库路径：")
print(db.db_path)
print()


# -------------------------
# 7. 批量生成并存储 Hamiltonian
# -------------------------
print("【步骤 7】批量生成并存储 Hamiltonian...")

num_to_generate = 3
inserted = generate_and_store_batch(
    gen,
    db,
    n=num_to_generate,
    seed_start=0,
)

print(f"请求生成 {num_to_generate} 条数据。")
print(f"实际插入数据库的数量：{inserted}")
print()


# -------------------------
# 8. 从数据库加载 Hamiltonian
# -------------------------
print("【步骤 8】从数据库中加载 Hamiltonian...")

seed_load = 0
H_loaded = db.load(seed_load)

if H_loaded is not None:
    print(f"成功从数据库加载 seed = {seed_load} 的 Hamiltonian ✔")
    print("Hamiltonian 类型：", type(H_loaded))
    print("Hilbert 空间维数：", H_loaded.Ns)
else:
    print("错误：未能从数据库加载 Hamiltonian ✘")

print()


# -------------------------
# 9. 总结
# -------------------------
print("所有测试步骤执行完毕。")
print("如果以上步骤均无报错，说明模块可以在其他 .py 文件中正常复用。")
