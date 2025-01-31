#Synthetic experiments (Sec 5.1-5.2).

import numpy as np
from core.synchronization import QuantumSyncQUBO
from core.annealing import QuantumAnnealer
from utils.metrics import hamming_accuracy

def synthetic_experiment(n=3, m=4, noise_level=0.1):
    # Generate synthetic pairwise permutations
    X_gt = [np.eye(n) for _ in range(m)]  # Ground truth (identity)
    pairwise_mats = []
    for i in range(m):
        for j in range(m):
            if i != j:
                # Add noise by swapping rows
                P_noisy = X_gt[i] @ X_gt[j].T
                swaps = np.random.choice(n, int(noise_level * n), replace=False)
                P_noisy[swaps] = P_noisy[np.random.permutation(swaps)]
                pairwise_mats.append(P_noisy)
    
    # Build and solve QUBO
    qubo_builder = QuantumSyncQUBO(pairwise_mats, lambda_reg=2.5)
    Q = qubo_builder.build_qubo()
    annealer = QuantumAnnealer(chain_strength=3.0)
    solution = annealer.solve(Q)
    
    # Evaluate accuracy
    solution_mat = solution.reshape((m, n, n))
    accuracy = hamming_accuracy(solution_mat, X_gt)
    print(f"Accuracy: {accuracy:.2f}")
