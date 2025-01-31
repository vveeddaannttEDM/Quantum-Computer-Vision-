import numpy as np

class PermutationConstraints:
    def __init__(self, n):
        self.n = n  # Size of permutation matrix (n x n)

    def matrix(self):
        """Constraint matrix for permutation structure (Eq. 4)."""
        A = np.zeros((2*self.n, self.n**2))
        for i in range(self.n):
            # Row constraints
            A[i, i*self.n : (i+1)*self.n] = 1
            # Column constraints
            A[self.n + i, i::self.n] = 1
        return A

    def vector(self):
        """Constraint vector (all ones)."""
        return np.ones(2 * self.n)
