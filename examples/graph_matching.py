import numpy as np
from core.frank_wolfe import QFW
from utils.constraints import PermutationConstraints
from utils.data_generator import generate_synthetic_matching

# Generate synthetic graph matching problem
n = 4  # Number of nodes
Q, A, b = generate_synthetic_matching(n)

# Solve with Q-FW
constraints = PermutationConstraints(n)
solver = QFW(Q, A, b, beta=1.5, max_iter=50)
solution = solver.optimize()

# Round to permutation matrix
permutation = solution.reshape((n, n))
permutation = np.round(permutation)  # Project to binary
print("Estimated permutation:\n", permutation)
