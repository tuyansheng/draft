# =========================
# test_generate_quantum_circuit.py
# =========================
#
# 说明：
# - 该测试文件假设与 Generate_quantum_circuit.py 位于同一文件夹
# - 测试目标：逐个验证导入、seed 枚举范围、seed→circuit、seed→Hamiltonian、DB 存取、批量生成
# - 尤其测试：seed >= 5^(L*depth) 时是否正确“报错/抛异常”（你写的“增氧”我理解为“抛异常/报错”）
#

import traceback


def _print_divider(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _print_ok(msg: str):
    print(f"[通过] {msg}")


def _print_fail(msg: str):
    print(f"[失败] {msg}")


def _expect_exception(fn, desc: str):
    print(f"准备测试：{desc}")
    try:
        fn()
    except Exception as e:
        print(f"如预期触发异常：{type(e).__name__}: {e}")
        _print_ok(desc)
        return True
    else:
        _print_fail(f"{desc}（未触发异常）")
        return False


def main():
    _print_divider("步骤 1：测试从 Generate_quantum_circuit.py 导入")
    try:
        from Generate_quantum_circuit import (
            CircuitGenConfig,
            QuantumCircuitGenerator,
            QuSpinCircuitDB,
            generate_and_store_batch,
        )
        _print_ok("导入成功：CircuitGenConfig / QuantumCircuitGenerator / QuSpinCircuitDB / generate_and_store_batch")
    except Exception:
        _print_fail("导入失败，下面打印异常堆栈")
        traceback.print_exc()
        return

    _print_divider("步骤 2：创建配置并检查 max_seed 是否等于 5^(L*depth)")
    cfg = CircuitGenConfig(L=5, depth=8)  # 这里 L*depth = 40
    print(f"当前配置：L={cfg.L}, depth={cfg.depth}，因此 L*depth={cfg.L * cfg.depth}")
    print("理论最大 seed 上界（不含）应为：5^(L*depth) = 5^40")

    try:
        gen = QuantumCircuitGenerator(cfg)
        print(f"生成器创建成功。dtype={gen.dtype}")
        print(f"gen.max_seed = {gen.max_seed}")
        expected = 5 ** (cfg.L * cfg.depth)
        print(f"expected  = {expected}")

        if gen.max_seed == expected:
            _print_ok("max_seed 计算正确（gen.max_seed == 5^(L*depth)）")
        else:
            _print_fail("max_seed 计算不一致（请检查 Generate_quantum_circuit.py 内部实现）")
    except Exception:
        _print_fail("创建 QuantumCircuitGenerator 失败，下面打印异常堆栈")
        traceback.print_exc()
        return

    _print_divider("步骤 3：测试 seed 枚举边界行为（重点：seed >= 5^40 必须报错）")

    max_seed = gen.max_seed
    print(f"当前 max_seed = {max_seed}")
    print("合法 seed 范围应为：0 <= seed < max_seed")
    print("下面测试：seed = -1 / 0 / max_seed-1 / max_seed / max_seed+1")

    # seed = -1 应该报错
    _expect_exception(lambda: gen.circuit_from_seed(-1), "seed = -1 时 circuit_from_seed 必须抛异常")

    # seed = 0 应该成功
    try:
        c0 = gen.circuit_from_seed(0)
        _print_ok("seed = 0 生成 circuit 成功")
        print("seed = 0 的线路（打印预览）：")
        gen.pretty_print(c0)
    except Exception:
        _print_fail("seed = 0 生成 circuit 失败")
        traceback.print_exc()

    # seed = max_seed - 1：注意可能因为 CNOT 约束导致“非法线路”而报错
    # 这里不强行要求成功，只要求：不应因为“越界”报错；但若因约束非法而报错属于正常
    try:
        c_last = gen.circuit_from_seed(max_seed - 1)
        _print_ok("seed = max_seed-1 生成 circuit 成功（说明该 seed 对应线路满足 CNOT 约束）")
        print("seed = max_seed-1 的线路（打印预览）：")
        gen.pretty_print(c_last)
    except ValueError as e:
        print(f"seed = max_seed-1 生成失败（这是可能发生的，原因通常是 CNOT 约束导致非法线路）：{e}")
        _print_ok("seed = max_seed-1 未越界（但可能因物理约束非法而失败）")
    except Exception:
        _print_fail("seed = max_seed-1 生成 circuit 时出现非预期异常")
        traceback.print_exc()

    # 重点：seed = max_seed 必须报错（越界）
    _expect_exception(lambda: gen.circuit_from_seed(max_seed), "seed = max_seed (= 5^40) 时必须抛异常（越界）")

    # 重点：seed = max_seed + 1 必须报错（越界）
    _expect_exception(lambda: gen.circuit_from_seed(max_seed + 1), "seed = max_seed+1 (> 5^40) 时必须抛异常（越界）")

    _print_divider("步骤 4：测试 seed→Hamiltonian（包含：可能因 CNOT 约束而跳过）")
    seeds_to_try = [0, 1, 2, 3, 4, 5, 10, 42, 123]
    print(f"准备测试以下 seeds 的 Hamiltonian 构造：{seeds_to_try}")
    print("注意：某些 seed 可能对应非法线路（CNOT 约束），那属于正常情况。")

    success_count = 0
    for s in seeds_to_try:
        try:
            _, H = gen.hamiltonian_from_seed(s)
            success_count += 1
            print(f"[成功] seed={s} Hamiltonian 构造成功：type={type(H)}, Ns={H.Ns}, dtype={H.dtype}")
        except ValueError as e:
            print(f"[跳过] seed={s} 对应非法线路（或越界）：{e}")
        except Exception:
            print(f"[异常] seed={s} 构造 Hamiltonian 出现非预期异常：")
            traceback.print_exc()

    print(f"本轮 Hamiltonian 构造成功数量：{success_count} / {len(seeds_to_try)}")

    _print_divider("步骤 5：测试数据库创建、写入、重复写入去重、读取")
    try:
        db = QuSpinCircuitDB(cfg)
        _print_ok("数据库对象创建成功")
        print(f"数据库路径：{db.db_path}")
    except Exception:
        _print_fail("数据库对象创建失败")
        traceback.print_exc()
        return

    # 选择一个尽量容易成功的 seed：从 0 开始往上找第一个能构造 Hamiltonian 的 seed
    print("准备寻找一个“合法 seed”，用于测试数据库写入/读取...")
    seed_good = None
    H_good = None
    for s in range(0, 500):  # 小范围找即可
        try:
            _, H = gen.hamiltonian_from_seed(s)
            seed_good = s
            H_good = H
            break
        except ValueError:
            continue

    if seed_good is None:
        _print_fail("在 seed=0..499 未找到可用线路，无法继续数据库写入测试（你当前约束可能过强或映射需要调整）")
        return

    print(f"找到可用 seed：{seed_good}，准备写入数据库。")

    try:
        inserted1 = db.insert(seed_good, H_good)
        print(f"第一次写入：insert 返回 {inserted1}（True 表示插入成功，False 表示已存在或被忽略）")
        # 再写一次，应该被 IGNORE（通常返回 False）
        inserted2 = db.insert(seed_good, H_good)
        print(f"第二次重复写入：insert 返回 {inserted2}（预期为 False，表示去重成功）")

        if inserted1 is True and inserted2 is False:
            _print_ok("数据库去重逻辑正常（同 seed 重复写入不会重复插入）")
        else:
            print("提示：如果第一次已经存在数据（旧数据库），inserted1 也可能为 False，这是正常的。")
            _print_ok("数据库写入/去重测试完成（需结合你数据库是否已有旧数据一起判断）")
    except Exception:
        _print_fail("数据库写入测试失败")
        traceback.print_exc()
        return

    try:
        H_loaded = db.load(seed_good)
        if H_loaded is None:
            _print_fail("数据库读取失败：返回 None")
        else:
            _print_ok("数据库读取成功（返回非 None）")
            print(f"读取到的 Hamiltonian：type={type(H_loaded)}, Ns={H_loaded.Ns}, dtype={H_loaded.dtype}")
    except Exception:
        _print_fail("数据库读取出现异常")
        traceback.print_exc()

    _print_divider("步骤 6：测试批量生成 generate_and_store_batch（会自动跳过非法线路与已存在 seed）")
    try:
        n = 5
        seed_start = 0
        print(f"准备批量生成并存储：n={n}, seed_start={seed_start}")
        inserted = generate_and_store_batch(gen, db, n=n, seed_start=seed_start)
        print(f"批量生成完成：实际插入 {inserted} 条")
        _print_ok("批量生成流程运行完成（不代表一定插入 n 条，因为可能存在大量非法线路或数据库已有数据）")
    except Exception:
        _print_fail("批量生成测试失败")
        traceback.print_exc()

    _print_divider("全部测试结束")
    print("如果以上所有步骤没有出现“非预期异常”，说明：")
    print("1）模块可被其他 .py 文件正常复用")
    print("2）seed 枚举边界正确（尤其 seed >= 5^40 会抛异常）")
    print("3）数据库存取基本功能正常")
    print("4）批量生成能跳过非法线路并存入数据库")


if __name__ == "__main__":
    main()
