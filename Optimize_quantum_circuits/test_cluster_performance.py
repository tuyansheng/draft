# ============================================================
# test_cluster_performance.py
# ============================================================

import os
import time

from Generate_quantum_circuit import CircuitGenConfig
from circuit_classifier import QuantumCircuitClassifier

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DATA_DIR = "/home/ystu/research_code/Machine_Learning/Optimize_quantum_circuits/data"
os.makedirs(DATA_DIR, exist_ok=True)

H5_PATH = os.path.join(DATA_DIR, "quantum_circuit_dataset.h5")

# circuit parameters (adjust if needed)
cfg = CircuitGenConfig(
    L=2,
    depth=4,
)

SEED_START = 0
SEED_STOP = 5**(cfg.L * cfg.depth)  # process all seeds

# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------
def main():
    print("=== Cluster Performance Test ===")
    print(f"Dataset path: {H5_PATH}")
    print(f"Seeds range: [{SEED_START}, {SEED_STOP})")
    print(f"Circuit config: L={cfg.L}, depth={cfg.depth}")
    print("--------------------------------")

    classifier = QuantumCircuitClassifier(H5_PATH, cfg)

    t0 = time.time()
    classifier.process_range(SEED_START, SEED_STOP)
    t1 = time.time()

    elapsed = t1 - t0

    print("--------------------------------")
    print("Computation finished.")
    print(f"Total seeds processed: {SEED_STOP - SEED_START}")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Average time per seed: {elapsed / (SEED_STOP - SEED_START):.6f} seconds")
    print("--------------------------------")

    # optional diagnostics
    classifier.print_equivalence_classes()
    classifier.print_illegal_seeds()
    classifier.print_hdf5_structure()


if __name__ == "__main__":
    main()
