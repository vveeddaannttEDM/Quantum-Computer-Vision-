## Quantum Permutation Synchronization

### Problem Formulation
Given a set of noisy pairwise permutation matrices \(\{\mathbf{P}_{ij}\}\) between \(m\) views, the goal is to recover consistent absolute permutation matrices \(\{\mathbf{X}_i\}\) by solving:

$
\min_{\{\mathbf{X}_i \in \mathcal{P}_n\}} \sum_{(i,j)\in\mathcal{E}} \|\mathbf{P}_{ij} - \mathbf{X}_i \mathbf{X}_j^\top\|_F^2,
$

where:
- \(\mathcal{P}_n\) is the set of \(n \times n\) permutation matrices.
- \(\mathcal{E}\) is the set of edges in the view graph.

### QUBO Construction
1. **Objective Term**:  
   The synchronization objective is rephrased as a quadratic unconstrained binary optimization (QUBO) problem:
   $$
   \mathbf{Q}_{\text{sync}} = -\sum_{(i,j)\in\mathcal{E}} \mathbf{I} \otimes \mathbf{P}_{ij},
   $$
   where \(\otimes\) denotes the Kronecker product.

2. **Permutation Constraints**:  
   Row/column constraints (\(\mathbf{X}_i \mathbf{1} = \mathbf{1}, \mathbf{X}_i^\top \mathbf{1} = \mathbf{1}\)) are encoded as quadratic penalties:
   $$
   \mathbf{Q}_{\text{constraint}} = \lambda \mathbf{A}^\top \mathbf{A},
   $$
   where \(\mathbf{A}\) is the constraint matrix and \(\lambda > 0\) is a regularization weight.

3. **Total QUBO**:  
   The final QUBO combines both terms:
   $$
   \mathbf{Q}_{\text{total}} = \mathbf{Q}_{\text{sync}} + \mathbf{Q}_{\text{constraint}}.
   $$

### Quantum Annealing
The QUBO is solved on a D-Wave quantum annealer by minimizing the Ising Hamiltonian:
$$
\mathcal{H} = \sum_{i,j} J_{ij} \sigma_i \sigma_j + \sum_i h_i \sigma_i,
$$
where \(\sigma_i \in \{-1, +1\}\) are qubit spins, \(J_{ij}\) are couplings, and \(h_i\) are biases. The ground state of \(\mathcal{H}\) corresponds to the optimal solution.

### Key Contributions
1. **Constraint Integration**: Permutation constraints are natively embedded into the QUBO, avoiding post-hoc rounding.
2. **Quantum Advantage**: Exploits quantum tunneling to escape local minima, improving solution quality over classical methods.
3. **Multi-View Consistency**: Synchronizes permutations across arbitrary graphs of views.

### Advantages
- **Global Optimality**: High probability of sampling the global optimum for small-to-medium problems.
- **Scalability**: Embeds problems on quantum hardware with up to \(n=5\) and \(m=8\) on current D-Wave devices.
- **Robustness**: Tolerates noisy/missing pairwise permutations via cycle consistency.# Quantum-Computer-Vision
The integration of Q-FW (Quantum Frank-Wolfe) and QuantumSync creates a hybrid classical-quantum framework for solving constrained quadratic binary optimization (QBO) problems, particularly permutation synchronization. Below is the unified theoretical explanation:
An affine transformation is a function between affine spaces that preserves collinearity


