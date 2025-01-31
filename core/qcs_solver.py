import numpy as np
from dwave.system import LeapHybridSampler
from .qubo_builder import QCSQUBOBuilder

class QCSSolver:
    def __init__(self, pairwise_mats, lambda_init=2.5, beta=1.0, max_iter=50):
        self.pairwise_mats = pairwise_mats
        self.lambda_reg = lambda_init
        self.beta = beta
        self.max_iter = max_iter
        self.sampler = LeapHybridSampler()
        self.qubo_builder = QCSQUBOBuilder(pairwise_mats)

    def optimize(self):
        n, m = self.pairwise_mats[0].shape[0], len(self.pairwise_mats)
        W = np.zeros((n+1, n+1))  # Copositive variable
        y = np.zeros(2 * n * m)   # Dual variables

        for t in range(self.max_iter):
            # Compute gradient with augmented Lagrangian
            grad = self._compute_gradient(W, y)
            
            # Build QUBO with adaptive constraints
            Q, linear = self.qubo_builder.build(grad, self.lambda_reg, y)
            
            # Solve on D-Wave
            response = self.sampler.sample_qubo(Q, linear=linear)
            direction = response.first.sample
            
            # Update primal and dual variables
            W = self._fw_update(W, direction, t)
            y = self._dual_update(y, W)
            
            # Adjust lambda and beta adaptively
            self._adapt_parameters(t)
        return self._round_solution(W)

    def _compute_gradient(self, W, y):
        # Objective gradient + Lagrangian terms
        return self.qubo_builder.sync_gradient(W) + self.lambda_reg * (self.qubo_builder.constraint_gradient(W) - y)

    def _adapt_parameters(self, t):
        # Rule: Increase lambda if constraints are violated, decrease beta
        self.lambda_reg *= 1.1
        self.beta /= np.sqrt(t + 1)
