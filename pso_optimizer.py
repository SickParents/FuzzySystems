import numpy as np
import softpy
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm
# from softpy.singlestate import MetaHeuristicsAlgorithm
from particle_candidate import ParticleCandidate

class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
    def __init__(self, fitness_func, pop_size, n_neighbors, **kwargs):
        super().__init__(fitness_func)
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

        self.population = []
        self.best = [None] * pop_size
        self.fitness_best = np.full(pop_size, -np.inf)

        self.global_best = None
        self.global_fitness_best = -np.inf

    def fit(self, n_iters):
        self.population = [
            ParticleCandidate.generate(**self.kwargs)
            for _ in range(self.pop_size)
        ]

        for i, particle in enumerate(self.population):
            fitness = self.fitness_function(particle)
            self.best[i] = particle.copy()
            self.fitness_best[i] = fitness

            if fitness > self.global_fitness_best:
                self.global_best = particle.copy()
                self.global_fitness_best = fitness

        for _ in range(n_iters):
            for i, particle in enumerate(self.population):
                fitness = self.fitness_function(particle)

                if fitness > self.fitness_best[i]:
                    self.fitness_best[i] = fitness
                    self.best[i] = particle.copy()

                if fitness > self.global_fitness_best:
                    self.global_fitness_best = fitness
                    self.global_best = particle.copy()

            for i, particle in enumerate(self.population):
                indices = [j for j in range(self.pop_size) if j != i]
                neighbors = np.random.choice(indices, self.n_neighbors, replace=False)

                best_neighbor_index = max(neighbors, key=lambda j: self.fitness_best[j])
                neighbor_best = self.best[best_neighbor_index]

                particle.recombine(self.best[i], neighbor_best, self.global_best)
                particle.mutate()
