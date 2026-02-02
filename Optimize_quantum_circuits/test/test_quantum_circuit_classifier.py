# ============================================================
# test_circuit_classifier_verbose.py
#
# 这是一个【独立可运行】的“超详细”测试脚本：
#   - 每处理一个 seed 都会输出一次信息
#   - 尽量覆盖 circuit_classifier.py 的每个功能点
#
# 使用方法：
#   1. 确保以下文件在同一目录：
#        - Generate_quantum_circuit.py
#        - circuit_classifier.py
#        - test_circuit_classifier_verbose.py（本文件）
#   2. 运行：
#        python test_circuit_classifier_verbose.py
#
# 建议测试参数：
#   - L=3, depth=2：空间不大，容易出现非法 seed，跑得快
#   - L=4, depth=2：更“有参考意义”，但计算更慢（可选）
# ============================================================

import os
import time
import h5py
import numpy as np

from Generate_quantum_circuit import CircuitGenConfig, QuantumCircuitGenerator
from circuit_classifier import QuantumCircuitClassifier, load_csr, csr_equal


# ------------------------------------------------------------
# 打印工具
# ------------------------------------------------------------
def banner(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def sub(title: str):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


def now():
    return time.strftime("%H:%M:%S")


def show_h5_overview(h5_path: str, max_show_classes: int = 6, max_show_seeds: int = 12):
    with h5py.File(h5_path, "r") as f:
        print(f"[{now()}] 【HDF5 顶层 keys】{list(f.keys())}")
        print(f"[{now()}] 【meta】L={int(f['meta/L'][()])}, depth={int(f['meta/depth'][()])}")

        all_seeds = f["seed_index/all_seeds"][:]
        illegal = f["illegal/seeds"][:]
        eq = f["equivalence_classes"]

        print(f"[{now()}] 【seed_index/all_seeds】数量={len(all_seeds)}，前{min(max_show_seeds,len(all_seeds))}个={all_seeds[:max_show_seeds]}")
        print(f"[{now()}] 【illegal/seeds】数量={len(illegal)}，前{min(max_show_seeds,len(illegal))}个={illegal[:max_show_seeds]}")
        print(f"[{now()}] 【equivalence_classes】类数量={len(eq)}")

        names = list(eq.keys())
        for name in names[:max_show_classes]:
            seeds = eq[f"{name}/seeds"][:]
            shape = tuple(eq[f"{name}/matrix/shape"][:])
            nnz = len(eq[f"{name}/matrix/data"][:])
            print(f"[{now()}]   - {name}: seeds数量={len(seeds)}, shape={shape}, nnz={nnz}, seeds前几个={seeds[:min(max_show_seeds,len(seeds))]}")


def get_class_id_of_seed(h5_path: str, seed: int):
    with h5py.File(h5_path, "r") as f:
        for cls_name in f["equivalence_classes"]:
            seeds = f[f"equivalence_classes/{cls_name}/seeds"][:]
            if seed in seeds:
                return cls_name
    return None


def in_illegal(h5_path: str, seed: int) -> bool:
    with h5py.File(h5_path, "r") as f:
        return seed in f["illegal/seeds"][:]


def in_all_seeds(h5_path: str, seed: int) -> bool:
    with h5py.File(h5_path, "r") as f:
        return seed in f["seed_index/all_seeds"][:]


# ------------------------------------------------------------
# 逐个 seed 处理并输出详细信息
# ------------------------------------------------------------
def process_one_seed_verbose(clf: QuantumCircuitClassifier, seed: int, note: str = ""):
    h5_path = clf.h5_path

    existed_before = in_all_seeds(h5_path, seed)
    cls_before = get_class_id_of_seed(h5_path, seed)
    illegal_before = in_illegal(h5_path, seed)

    print(f"[{now()}] 处理 seed={seed} {('('+note+')') if note else ''}")
    print(f"[{now()}]   - 处理前：已存在于 all_seeds？{existed_before}")
    print(f"[{now()}]   - 处理前：是否非法？{illegal_before}")
    print(f"[{now()}]   - 处理前：所在等价类={cls_before}")

    # 记录处理前计数
    with h5py.File(h5_path, "r") as f:
        n_total_before = len(f["seed_index/all_seeds"])
        n_illegal_before = len(f["illegal/seeds"])
        n_classes_before = len(f["equivalence_classes"])

    t0 = time.time()
    clf.process_seed(seed)
    t1 = time.time()

    existed_after = in_all_seeds(h5_path, seed)
    cls_after = get_class_id_of_seed(h5_path, seed)
    illegal_after = in_illegal(h5_path, seed)

    # 记录处理后计数
    with h5py.File(h5_path, "r") as f:
        n_total_after = len(f["seed_index/all_seeds"])
        n_illegal_after = len(f["illegal/seeds"])
        n_classes_after = len(f["equivalence_classes"])

    print(f"[{now()}]   - process_seed 用时 {t1 - t0:.4f} 秒")
    print(f"[{now()}]   - 处理后：已存在于 all_seeds？{existed_after}")
    print(f"[{now()}]   - 处理后：是否非法？{illegal_after}")
    print(f"[{now()}]   - 处理后：所在等价类={cls_after}")
    print(f"[{now()}]   - 计数变化：all_seeds {n_total_before}->{n_total_after}，illegal {n_illegal_before}->{n_illegal_after}，classes {n_classes_before}->{n_classes_after}")

    # 推断发生了什么
    if existed_before:
        print(f"[{now()}]   - 结论：seed 之前已处理，本次应当无任何写入（去重路径）。")
    else:
        if illegal_after:
            print(f"[{now()}]   - 结论：seed 被判定为非法，已写入 illegal/seeds。")
        else:
            if n_classes_after > n_classes_before:
                print(f"[{now()}]   - 结论：产生了新等价类（新矩阵），并保存了该 seed。")
            else:
                print(f"[{now()}]   - 结论：命中了已有等价类（矩阵完全相同），seed 被追加进该类。")


# ------------------------------------------------------------
# 自动挑选一些“有参考意义”的 seed
# ------------------------------------------------------------
def pick_legal_and_illegal_seeds(cfg: CircuitGenConfig, want_legal: int = 12, want_illegal: int = 8, scan_limit: int = 4000):
    gen = QuantumCircuitGenerator(cfg)
    legal, illegal = [], []

    for s in range(min(gen.max_seed, scan_limit)):
        try:
            gen.circuit_from_seed(s)
            if len(legal) < want_legal:
                legal.append(s)
        except Exception:
            if len(illegal) < want_illegal:
                illegal.append(s)

        if len(legal) >= want_legal and len(illegal) >= want_illegal:
            break

    return legal, illegal


def find_equivalent_pair(cfg: CircuitGenConfig, scan_limit: int = 3000):
    """
    在一定范围内扫描，尝试找到两个不同 seed 产生完全相同的稀疏矩阵（等价碰撞）。
    找不到不算失败：因为某些参数/范围下确实可能没有。
    """
    gen = QuantumCircuitGenerator(cfg)
    seen = {}  # seed -> csr

    for s in range(scan_limit):
        try:
            _, H = gen.hamiltonian_from_seed(s)
            Hc = H.tocsr()
        except Exception:
            continue

        for s0, H0 in seen.items():
            if csr_equal(H0, Hc):
                return (s0, s)
        seen[s] = Hc

    return None


# ------------------------------------------------------------
# 测试 1：初始化与 HDF5 结构
# ------------------------------------------------------------
def test_initialization(h5_path: str, cfg: CircuitGenConfig):
    banner("测试 1：初始化 QuantumCircuitClassifier 与 HDF5 结构创建/检查")

    if os.path.exists(h5_path):
        os.remove(h5_path)
        print(f"[{now()}] 已删除旧文件：{h5_path}")
    else:
        print(f"[{now()}] 未发现旧文件，将创建：{h5_path}")

    clf = QuantumCircuitClassifier(h5_path, cfg)

    with h5py.File(h5_path, "r") as f:
        print(f"[{now()}] HDF5 顶层结构：{list(f.keys())}")
        assert "meta" in f
        assert "equivalence_classes" in f
        assert "illegal" in f
        assert "seed_index" in f

        L = int(f["meta/L"][()])
        depth = int(f["meta/depth"][()])
        print(f"[{now()}] meta/L={L}, meta/depth={depth}")
        assert (L, depth) == (cfg.L, cfg.depth)

        assert "seeds" in f["illegal"]
        assert "all_seeds" in f["seed_index"]

    print(f"[{now()}] 初始化测试通过")
    show_h5_overview(h5_path)
    return clf


# ------------------------------------------------------------
# 测试 2：单 seed 处理（合法/非法/越界/重复）
# ------------------------------------------------------------
def test_single_seed_paths(clf: QuantumCircuitClassifier):
    banner("测试 2：逐条 seed 测试（合法/非法/越界/重复）")

    cfg = clf.cfg
    legal, illegal = pick_legal_and_illegal_seeds(cfg, want_legal=6, want_illegal=4, scan_limit=2000)

    sub("2.1 选取若干合法 seed，逐个处理并详细输出")
    for s in legal:
        process_one_seed_verbose(clf, s, note="合法候选")

    sub("2.2 选取若干非法 seed，逐个处理并详细输出")
    for s in illegal:
        process_one_seed_verbose(clf, s, note="非法候选")

    sub("2.3 越界 seed 测试（max_seed）")
    gen = QuantumCircuitGenerator(cfg)
    oob = gen.max_seed  # 越界：合法范围是 [0, max_seed)
    process_one_seed_verbose(clf, oob, note="越界（应当进入 illegal）")

    sub("2.4 重复 seed 测试（去重路径）")
    if legal:
        s = legal[0]
        process_one_seed_verbose(clf, s, note="重复处理（应去重，不写入）")

    print(f"[{now()}] 单 seed 路径测试完成")
    show_h5_overview(clf.h5_path, max_show_classes=10)


# ------------------------------------------------------------
# 测试 3：批量接口 process_seeds
# ------------------------------------------------------------
def test_batch_process_seeds(clf: QuantumCircuitClassifier):
    banner("测试 3：批量接口 process_seeds（逐个 seed 输出 + 批量调用后检查）")

    cfg = clf.cfg
    # 选择一批连续 seed：同时包含合法和非法的概率更高
    seeds = list(range(0, 40))

    sub("3.1 批量前：逐个 seed 预检查（是否已处理）")
    for s in seeds:
        print(f"[{now()}] 预检查 seed={s}，已处理？{in_all_seeds(clf.h5_path, s)}")

    sub("3.2 调用 clf.process_seeds(seeds)")
    t0 = time.time()
    clf.process_seeds(seeds)
    t1 = time.time()
    print(f"[{now()}] process_seeds 总用时 {t1 - t0:.4f} 秒，数量={len(seeds)}")

    sub("3.3 批量后：逐个 seed 后验检查（归属非法/等价类）")
    for s in seeds:
        if in_illegal(clf.h5_path, s):
            print(f"[{now()}] seed={s} -> 非法")
        else:
            cls = get_class_id_of_seed(clf.h5_path, s)
            if cls is None:
                # 理论上不应该发生：只要写入 all_seeds，就应该在非法或某个类
                print(f"[{now()}] seed={s} -> 【异常】不在 illegal 也不在任何等价类")
            else:
                print(f"[{now()}] seed={s} -> 等价类 {cls}")

    print(f"[{now()}] 批量接口 process_seeds 测试完成")
    show_h5_overview(clf.h5_path, max_show_classes=12)


# ------------------------------------------------------------
# 测试 4：批量接口 process_range
# ------------------------------------------------------------
def test_batch_process_range(clf: QuantumCircuitClassifier):
    banner("测试 4：批量接口 process_range（并验证与逐 seed 一致性）")

    start, stop = 40, 70
    print(f"[{now()}] 准备调用 process_range({start}, {stop})")

    # range 前快照
    with h5py.File(clf.h5_path, "r") as f:
        before = len(f["seed_index/all_seeds"])

    t0 = time.time()
    clf.process_range(start, stop)
    t1 = time.time()

    with h5py.File(clf.h5_path, "r") as f:
        after = len(f["seed_index/all_seeds"])

    print(f"[{now()}] process_range 用时 {t1 - t0:.4f} 秒")
    print(f"[{now()}] all_seeds 数量变化：{before} -> {after}（新增最多 {stop-start} 条，因去重可能更少）")

    sub("4.1 对区间内每个 seed 输出归属（非法/类）")
    for s in range(start, stop):
        if in_illegal(clf.h5_path, s):
            print(f"[{now()}] seed={s} -> 非法")
        else:
            cls = get_class_id_of_seed(clf.h5_path, s)
            print(f"[{now()}] seed={s} -> 等价类 {cls}")

    print(f"[{now()}] process_range 测试完成")
    show_h5_overview(clf.h5_path, max_show_classes=12)


# ------------------------------------------------------------
# 测试 5：等价类聚合（尽量尝试找到等价 seed 对）
# ------------------------------------------------------------
def test_equivalence_merge(clf: QuantumCircuitClassifier):
    banner("测试 5：等价类聚合（不同 seed -> 完全相同稀疏矩阵）")

    cfg = clf.cfg
    pair = find_equivalent_pair(cfg, scan_limit=3000)

    if pair is None:
        print(f"[{now()}] 在 scan_limit=3000 范围内未找到等价 seed 对。")
        print(f"[{now()}] 这不一定是 bug：可能碰撞很少。你可以增大 scan_limit 或增大扫描范围。")
        return

    s1, s2 = pair
    print(f"[{now()}] 找到等价 seed 对：s1={s1}, s2={s2}")

    sub("5.1 逐个处理这两个 seed，并观察是否落入同一等价类")
    process_one_seed_verbose(clf, s1, note="等价对 seed1")
    process_one_seed_verbose(clf, s2, note="等价对 seed2")

    c1 = get_class_id_of_seed(clf.h5_path, s1)
    c2 = get_class_id_of_seed(clf.h5_path, s2)
    print(f"[{now()}] seed={s1} 所在类={c1}")
    print(f"[{now()}] seed={s2} 所在类={c2}")

    assert c1 is not None and c2 is not None
    assert c1 == c2

    with h5py.File(clf.h5_path, "r") as f:
        seeds_in = f[f"equivalence_classes/{c1}/seeds"][:]
    print(f"[{now()}] 等价类 {c1} 当前 seeds 数量={len(seeds_in)}，前若干={seeds_in[:min(20,len(seeds_in))]}")
    print(f"[{now()}] 等价类聚合测试通过")


# ------------------------------------------------------------
# 测试 6：矩阵一致性（HDF5 内矩阵 vs 重新生成）
# ------------------------------------------------------------
def test_matrix_roundtrip_consistency(clf: QuantumCircuitClassifier):
    banner("测试 6：矩阵一致性核对（从 HDF5 读回矩阵，与重新生成矩阵完全一致）")

    h5_path = clf.h5_path
    cfg = clf.cfg
    gen = QuantumCircuitGenerator(cfg)

    with h5py.File(h5_path, "r") as f:
        eq_names = list(f["equivalence_classes"].keys())
        if len(eq_names) == 0:
            print(f"[{now()}] 当前没有任何等价类，跳过该测试。")
            return

        cls_name = eq_names[0]
        seeds = f[f"equivalence_classes/{cls_name}/seeds"][:]
        if len(seeds) == 0:
            print(f"[{now()}] {cls_name} seeds 为空（不应发生），跳过。")
            return

        seed0 = int(seeds[0])
        mat_saved = load_csr(f[f"equivalence_classes/{cls_name}/matrix"])

    print(f"[{now()}] 选取等价类 {cls_name} 的 seed0={seed0} 做一致性核对")
    _, H = gen.hamiltonian_from_seed(seed0)
    mat_new = H.tocsr()

    ok = csr_equal(mat_saved, mat_new)
    print(f"[{now()}] 矩阵完全一致？{ok}")
    assert ok
    print(f"[{now()}] 矩阵一致性测试通过")


# ------------------------------------------------------------
# 测试 7：summary 接口（详细打印）
# ------------------------------------------------------------
def test_summary(clf: QuantumCircuitClassifier):
    banner("测试 7：summary 接口")

    s = clf.summary()
    print(f"[{now()}] summary 输出如下：")
    for k in sorted(s.keys()):
        print(f"[{now()}]   - {k}: {s[k]}")

    assert s["L"] == clf.cfg.L
    assert s["depth"] == clf.cfg.depth
    print(f"[{now()}] summary 测试通过")


# ------------------------------------------------------------
# 主入口：建议参数组合
# ------------------------------------------------------------
def main():
    banner("开始：超详细测试脚本")

    # -------------------------
    # 推荐测试参数（有参考意义、且运行快）
    # -------------------------
    cfg = CircuitGenConfig(L=3, depth=2)
    h5_path = "test_db_verbose_L3D2.h5"

    print(f"[{now()}] 本次测试参数：L={cfg.L}, depth={cfg.depth}")
    print(f"[{now()}] 本次测试数据库：{h5_path}")

    clf = test_initialization(h5_path, cfg)

    test_single_seed_paths(clf)
    test_batch_process_seeds(clf)
    test_batch_process_range(clf)
    test_equivalence_merge(clf)
    test_matrix_roundtrip_consistency(clf)
    test_summary(clf)

    banner("所有测试完成")
    print(f"[{now()}] 你可以用 HDFView / h5py 打开 {h5_path} 查看内部结构与数据。")
    print(f"[{now()}] 若要更“重”的测试：把 cfg 改成 L=4, depth=2 或把扫描范围加大。")


if __name__ == "__main__":
    main()
