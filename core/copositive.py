import numpy as np

class CopositiveReformulation:
    def __init__(self, n_vars, constraints):
        self.n = n_vars            # Number of binary variables
        self.m = constraints.shape[0]  # Number of constraints
        self.delta = []            # Copositive cone parameters

    def lift_to_copositive(self, Q, s):
        """Reformulate QBO as a copositive program (Eq. 7-8)."""
        p = self.n + 1
        C = np.block([
            [0, s.T],
            [s, Q]
        ])  # Combined cost matrix
        return C

    def build_constraints(self, A, b):
        """Construct affine constraints for copositive program."""
        # Ensure diag(X) = x and linear constraints (Eq. 7)
        constraints = []
        for i in range(self.m):
            Ai = np.outer(A[i], A[i])
            constraints.append((Ai, b[i]**2))
        return constraints
