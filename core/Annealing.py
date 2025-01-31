#Solves the QUBO on D-Wave and processes results.


from dwave.system import LeapHybridSampler, EmbeddingComposite
import numpy as np

class QuantumAnnealer:
    def __init__(self, chain_strength=3.0, num_reads=1000):
        self.sampler = EmbeddingComposite(LeapHybridSampler())
        self.chain_strength = chain_strength
        self.num_reads = num_reads
    
    def solve(self, qubo_matrix):
        """Solve QUBO on D-Wave and return the best solution."""
        response = self.sampler.sample_qubo(
            qubo_matrix,
            chain_strength=self.chain_strength,
            num_reads=self.num_reads
        )
        best_sample = response.first.sample
        return np.array(list(best_sample.values())).astype(int)
