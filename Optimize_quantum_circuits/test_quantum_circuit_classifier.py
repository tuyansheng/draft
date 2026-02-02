# ============================================================
# test_classifier_full.py
# ============================================================

import numpy as np
from scipy.sparse import csr_matrix

from quantum_circuit_classifier import QuantumCircuitClassifier


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_initialization():
    print_header("测试 1：初始化分类器")

    clf = QuantumCircuitClassifier(L=3, depth=2)
    print("初始化成功")
    print(f"L = {clf.dataset.L}, depth = {clf.dataset.depth}")
    print(f"初始等价类数量 = {clf.dataset.num_classes()}")
    print(f"初始非法 seed 数量 = {clf.dataset.total_invalid_circuits()}")
    return clf


def test_single_seed(clf):
    print_header("测试 2：单个 seed 分类（合法 + 非法）")

    test_seeds = [0, 1, 2, -1, 999999]

    for s in test_seeds:
        cid = clf.classify_seed(s)
        if cid is None:
            print(f"seed {s} -> 非法线路")
        else:
            print(f"seed {s} -> 等价类 {cid}")

    clf.print_dataset_summary()


def test_batch_classification(clf):
    print_header("测试 3：批量 seed 分类")

    seeds = list(range(20)) + [1000, 2000, -5]
    result = clf.classify_seeds(seeds, verbose=True)

    print("\n批量分类结果：")
    for s, cid in result.items():
        print(f"  seed {s} -> {cid}")

    clf.print_dataset_summary()


def test_equivalence_class_details(clf):
    print_header("测试 4：等价类详细检查")

    if clf.dataset.num_classes() == 0:
        print("没有合法等价类")
        return

    for idx in range(clf.dataset.num_classes()):
        clf.print_class_info(idx)


def test_find_class_by_seed(clf):
    print_header("测试 5：按 seed 查找等价类")

    test_seeds = []
    if clf.dataset.classes:
        test_seeds.append(clf.dataset.classes[0].seeds[0])
    if clf.dataset.invalid_seeds:
        test_seeds.append(clf.dataset.invalid_seeds[0])
    test_seeds.append(123456789)

    for s in test_seeds:
        cid = clf.find_class_by_seed(s)
        print(f"seed {s} -> {cid}")


def test_hdf5_save(clf):
    print_header("测试 6：HDF5 保存")

    clf.save_to_hdf5("test_dataset.h5")
    print("保存完成")


def test_hdf5_load_and_verify():
    print_header("测试 7：HDF5 加载与一致性验证")

    clf = QuantumCircuitClassifier(L=1, depth=1)  # dummy
    clf.load_from_hdf5("test_dataset.h5")

    print("\n加载后数据集摘要：")
    clf.print_dataset_summary()

    # 检查矩阵一致性
    print("\n验证矩阵一致性：")
    for idx, cls in enumerate(clf.dataset.classes):
        mat = cls.representative_matrix
        if not isinstance(mat, csr_matrix):
            print(f"错误：class_{idx} 的矩阵不是 CSR 格式")
        else:
            print(f"class_{idx} 矩阵 OK，形状 = {mat.shape}")

    return clf


def test_cross_instance_consistency(clf1, clf2):
    print_header("测试 8：跨实例一致性验证")

    print(f"实例 1 等价类数量 = {clf1.dataset.num_classes()}")
    print(f"实例 2 等价类数量 = {clf2.dataset.num_classes()}")

    if clf1.dataset.num_classes() != clf2.dataset.num_classes():
        print("错误：等价类数量不一致")
        return

    for i in range(clf1.dataset.num_classes()):
        seeds1 = set(clf1.dataset.classes[i].seeds)
        seeds2 = set(clf2.dataset.classes[i].seeds)
        if seeds1 != seeds2:
            print(f"错误：class_{i} 的 seed 列表不一致")
        else:
            print(f"class_{i} 的 seed 列表一致")

    print("跨实例一致性验证完成")


def main():
    print_header("量子线路分类器全功能测试开始")

    clf = test_initialization()
    test_single_seed(clf)
    test_batch_classification(clf)
    test_equivalence_class_details(clf)
    test_find_class_by_seed(clf)
    test_hdf5_save(clf)

    clf_loaded = test_hdf5_load_and_verify()
    test_cross_instance_consistency(clf, clf_loaded)

    print_header("所有测试完成")


if __name__ == "__main__":
    main()
