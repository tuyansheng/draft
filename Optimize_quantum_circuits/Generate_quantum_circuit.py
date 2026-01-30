import json
import pickle
import random
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


Gate = str
Circuit2D = List[List[Gate]]


def _validate_circuit(circuit: Circuit2D) -> Tuple[int, int]:
    if not circuit or not isinstance(circuit, list) or not all(isinstance(r, list) for r in circuit):
        raise ValueError("circuit 必须是二维 list。")

    L = len(circuit)
    T = len(circuit[0]) if L > 0 else 0
    if T == 0:
        raise ValueError("circuit 的列数（深度）必须 > 0。")
    if any(len(r) != T for r in circuit):
        raise ValueError("circuit 每一行长度必须一致。")

    allowed = {"I", "X", "Y", "Z", "CNOT"}
    for q in range(L):
        for t in range(T):
            g = circuit[q][t]
            if not isinstance(g, str):
                raise ValueError(f"circuit[{q}][{t}] 必须是 str。")
            if g not in allowed:
                raise ValueError(f"非法门：circuit[{q}][{t}] = {g}，允许 {sorted(allowed)}。")

    # CNOT 约束：只能作用于 (q -> q+1)，且 target 行必须是 I，占位
    for q in range(L):
        for t in range(T):
            if circuit[q][t] == "CNOT":
                if q == L - 1:
                    raise ValueError(f"最后一行不能放 CNOT：circuit[{q}][{t}]。")
                if circuit[q + 1][t] != "I":
                    raise ValueError(
                        f"CNOT 的下一行同列必须为 I 占位：circuit[{q+1}][{t}] = {circuit[q+1][t]}"
                    )

    return L, T


def circuit_to_quspin_hamiltonian(
    circuit: Circuit2D,
    basis=None,
    dtype=float,
):
    """
    输入：二维列表线路（元素为 str）
    输出：quspin.operators.hamiltonian 类型
    """
    L, T = _validate_circuit(circuit)
    if basis is None:
        basis = spin_basis_1d(L)

    static = []

    # 单比特项
    for q in range(L):
        for t in range(T):
            g = circuit[q][t]
            if g == "X":
                static.append(["x", [[1.0, q]]])
            elif g == "Y":
                static.append(["y", [[1.0, q]]])
            elif g == "Z":
                static.append(["z", [[1.0, q]]])

    # CNOT 项：H = -(pi/4) Zc -(pi/4) Xt + (pi/4) ZcXt
    c = 3.141592653589793 / 4.0
    for q in range(L - 1):
        for t in range(T):
            if circuit[q][t] == "CNOT":
                static.append(["z", [[-c, q]]])          # -pi/4 Z_c
                static.append(["x", [[-c, q + 1]]])      # -pi/4 X_t
                static.append(["zx", [[+c, q, q + 1]]])  # +pi/4 Z_c X_t

    # 合并同类项不是必须；quspin 会处理列表形式的构造
    H = hamiltonian(static, [], basis=basis, dtype=dtype)
    return H


@dataclass
class CircuitGenConfig:
    depth: int
    p_cnot: float = 0.2
    p_identity: float = 0.4
    p_single_pauli: float = 0.4  # 总概率，内部均分 X/Y/Z


class QuSpinCircuitDB:
    """
    - 保存两类关键数据：二维列表（JSON）与 quspin Hamiltonian（pickle blob）
    - 提供：批量生成不重复线路；批量转 Hamiltonian 并写入库；打印可读线路
    """

    def __init__(self, L: int, db_path: str = "./circuits.db", dtype=float):
        if L <= 0:
            raise ValueError("L 必须 > 0。")
        self.L = L
        self.db_path = db_path
        self.dtype = dtype
        self.basis = spin_basis_1d(L)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS circuits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    circuit_json TEXT NOT NULL UNIQUE,
                    hamiltonian_blob BLOB
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_circuit_json ON circuits(circuit_json);")
            con.commit()

    @staticmethod
    def _canonical_json(circuit: Circuit2D) -> str:
        # ensure_ascii=False 方便中文环境；separators 压缩；sort_keys 保持稳定
        return json.dumps(circuit, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

    def _exists_in_db(self, circuit_json: str) -> bool:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT 1 FROM circuits WHERE circuit_json=? LIMIT 1;", (circuit_json,))
            return cur.fetchone() is not None

    def _random_gate(self, rng: random.Random, cfg: CircuitGenConfig) -> str:
        r = rng.random()
        if r < cfg.p_identity:
            return "I"
        r -= cfg.p_identity
        if r < cfg.p_single_pauli:
            return rng.choice(["X", "Y", "Z"])
        return "CNOT"

    def generate_unique_circuits(
        self,
        n: int,
        cfg: CircuitGenConfig,
        seed: Optional[int] = None,
        max_tries: int = 200000,
    ) -> List[Circuit2D]:
        """
        批量生成二维列表，保证：
        - 不与数据库重复
        - 本批次内部也不重复
        """
        if n <= 0:
            return []
        if cfg.depth <= 0:
            raise ValueError("cfg.depth 必须 > 0。")
        if not (0.0 <= cfg.p_cnot <= 1.0):
            raise ValueError("cfg.p_cnot 必须在 [0,1]。")

        # 概率规范化：p_identity + p_single_pauli + p_cnot = 1
        total = cfg.p_identity + cfg.p_single_pauli + cfg.p_cnot
        if abs(total - 1.0) > 1e-12:
            raise ValueError("概率需要满足 p_identity + p_single_pauli + p_cnot = 1。")

        rng = random.Random(seed)
        batch = []
        seen = set()

        tries = 0
        while len(batch) < n:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"生成失败：在 max_tries={max_tries} 内无法凑够 {n} 个不重复线路。"
                )

            # 先全置 I
            circuit = [["I" for _ in range(cfg.depth)] for _ in range(self.L)]

            # 逐列逐行随机填门，遇到 CNOT 则占用两行
            for t in range(cfg.depth):
                q = 0
                while q < self.L:
                    if q == self.L - 1:
                        # 最后一行禁 CNOT
                        g = rng.choice(["I", "X", "Y", "Z"])
                        circuit[q][t] = g
                        q += 1
                        continue

                    g = self._random_gate(rng, cfg)
                    if g == "CNOT":
                        circuit[q][t] = "CNOT"
                        circuit[q + 1][t] = "I"  # target 占位
                        q += 2
                    else:
                        circuit[q][t] = g
                        q += 1

            # 二次校验
            _validate_circuit(circuit)

            cj = self._canonical_json(circuit)
            if cj in seen:
                continue
            if self._exists_in_db(cj):
                continue

            seen.add(cj)
            batch.append(circuit)

        return batch

    def store_circuits_with_hamiltonians(
        self,
        circuits: List[Circuit2D],
    ) -> int:
        """
        将二维列表转 Hamiltonian，并把 (circuit_json, hamiltonian_blob) 存入 SQLite。
        返回：成功插入条数（若遇到 UNIQUE 冲突会跳过）。
        """
        if not circuits:
            return 0

        inserted = 0
        with sqlite3.connect(self.db_path) as con:
            for circuit in circuits:
                cj = self._canonical_json(circuit)
                if self._exists_in_db(cj):
                    continue

                H = circuit_to_quspin_hamiltonian(circuit, basis=self.basis, dtype=self.dtype)
                blob = pickle.dumps(H, protocol=pickle.HIGHEST_PROTOCOL)

                cur = con.execute(
                    """
                    INSERT OR IGNORE INTO circuits(circuit_json, hamiltonian_blob)
                    VALUES(?, ?);
                    """,
                    (cj, sqlite3.Binary(blob)),
                )
                # sqlite3 的 rowcount 对 OR IGNORE 一般可用；稳妥用查询也行
                if cur.rowcount == 1:
                    inserted += 1
            con.commit()

        return inserted

    def pretty_print(self, circuit: Circuit2D) -> None:
        """
        以“时间从左到右、qubit 从上到下”的形式打印。
        """
        L, T = _validate_circuit(circuit)

        # 列宽对齐
        def cell(g: str) -> str:
            return f"{g:>4}"

        header = "     " + "".join([f"{t:>4}" for t in range(T)])
        print(header)
        print("     " + "----" * T)

        for q in range(L):
            line = f"q{q:02d} " + "".join(cell(circuit[q][t]) for t in range(T))
            print(line)

    def load_hamiltonian_by_circuit(self, circuit: Circuit2D):
        """
        可选工具：通过二维列表从库中取回 Hamiltonian（若存在）。
        """
        cj = self._canonical_json(circuit)
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "SELECT hamiltonian_blob FROM circuits WHERE circuit_json=? LIMIT 1;",
                (cj,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            return pickle.loads(row[0])

if __name__ == "__main__":
    # =========================
    # 基本参数
    # =========================
    L = 5                 # qubit 数
    depth = 8             # 线路深度
    db_path = "./circuits.db"

    # =========================
    # 初始化数据库与管理类
    # =========================
    print(">>> 初始化 QuSpinCircuitDB")
    db = QuSpinCircuitDB(L=L, db_path=db_path)

    # =========================
    # 批量生成不重复二维线路
    # =========================
    print("\n>>> 批量生成二维线路")
    cfg = CircuitGenConfig(
        depth=depth,
        p_identity=0.4,
        p_single_pauli=0.4,
        p_cnot=0.2,
    )

    circuits = db.generate_unique_circuits(
        n=3,
        cfg=cfg,
        seed=42,
    )

    print(f"生成线路数量: {len(circuits)}")

    # =========================
    # 打印第一条线路（可读形式）
    # =========================
    print("\n>>> 打印第一条线路")
    db.pretty_print(circuits[0])

    # =========================
    # 转换为 Hamiltonian 并写入数据库
    # =========================
    print("\n>>> 写入数据库（线路 + Hamiltonian）")
    inserted = db.store_circuits_with_hamiltonians(circuits)
    print(f"成功写入条数: {inserted}")

    # =========================
    # 验证 Hamiltonian 类型
    # =========================
    print("\n>>> 从数据库中读取 Hamiltonian 并验证类型")
    H = db.load_hamiltonian_by_circuit(circuits[0])

    if H is not None:
        print("Hamiltonian 加载成功")
        print("类型:", type(H))
        print("Hilbert 空间维度:", H.Ns)
    else:
        print("Hamiltonian 未找到")

    # =========================
    # 再次生成，验证“无重复”逻辑
    # =========================
    print("\n>>> 再次生成线路，验证不会与数据库重复")
    circuits_new = db.generate_unique_circuits(
        n=2,
        cfg=cfg,
        seed=123,
    )

    inserted_new = db.store_circuits_with_hamiltonians(circuits_new)
    print(f"新增写入条数: {inserted_new}")

    print("\n>>> 所有功能验证完成")
