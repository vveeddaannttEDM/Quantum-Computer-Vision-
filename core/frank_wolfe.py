import numpy as np
from dwave.system import LeapHybridSampler
from .copositive import CopositiveReformulation

class QFW:
    def __init__(self, Q, A, b, beta=1.0, max_iter=100):
        self.Q = Q                  # Quadratic cost matrix
        self.A = A                  # Constraint matrix (Ax = b)
        self.b = b                  # Constraint vector
        self.beta = beta            # Penalty parameter
        self.max_iter = max_iter    # Max FW iterations
        self.sampler = LeapHybridSampler()  # D-Wave quantum sampler

    def _qubo_subproblem(self, gradient):
        """Solve QUBO: min_x x^T (Q + βA^TA) x + (s - βA^Tb)^T x"""
        qubo = self.Q + self.beta * self.A.T @ self.A
        linear = -2 * self.beta * self.A.T @ self.b
        return qubo, linear

    def _fw_step(self, x, y):
        """Frank-Wolfe iteration with QUBO direction."""
        # Compute gradient of the augmented Lagrangian
        grad = self.Q @ x + self.A.T @ y + self.beta * self.A.T @ (self.A @ x - self.b)
        
        # Build and solve QUBO
        qubo, linear = self._qubo_subproblem(grad)
        response = self.sampler.sample_qubo(qubo, linear=linear)
        direction = np.array(list(response.first.sample.values()))
        
        # Update primal variable
        step_size = 2 / (self.iter + 2)
        x_new = (1 - step_size) * x + step_size * direction
        return x_new

    def optimize(self):
        """Run Q-FW optimization loop."""
        x = np.zeros(self.Q.shape[0])  # Initial primal variable
        y = np.zeros(self.A.shape[0])  # Initial dual variable
        
        for iter in range(self.max_iter):
            x = self._fw_step(x, y)
            residual = self.A @ x - self.b
            y += self.beta * residual  # Dual ascent
        return x
      #Implements the Frank-Wolfe algorithm with QUBO subproblem solving.

