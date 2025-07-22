import numpy as np
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm
from particle_candidate import ParticleCandidate


class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
    '''
    Implementation of a Particle Swarm Optimization (PSO) algorithm.
    Compatible with the softpy framework and designed to work with ParticleCandidate objects.

    Parameters
    ----------
    :param fitness_func: the fitness function to be maximized
    :type fitness_func: Callable

    :param pop_size: number of particles in the swarm
    :type pop_size: int

    :param n_neighbors: number of neighbors considered for each particle
    :type n_neighbors: int

    :param kwargs: additional keyword arguments to pass to ParticleCandidate.generate()
    :type kwargs: dict
    '''
    def __init__(self, fitness_func, pop_size: int, n_neighbors: int, **kwargs):
        '''
        Constructor for the ParticleSwarmOptimizer class.
        Initializes population parameters and references to the fitness function.
        '''
        self.fitness_func = fitness_func
        # Initialize the MetaHeuristicsAlgorithm superclass with candidate type
        super().__init__(fitness_func=self.fitness_func, candidate_type=ParticleCandidate)
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

        # Will be populated during fit()
        self.population = []
        self.best = None
        self.fitness_best = None
        self.global_best = None
        self.global_fitness_best = None

    def fit(self, n_iters: int):
        '''
        Executes the Particle Swarm Optimization process.

        Parameters
        ----------
        :param n_iters: number of optimization iterations
        :type n_iters: int
        '''
        # --- Initialization phase ---
        # Generate initial population using ParticleCandidate factory
        self.population = [
            ParticleCandidate.generate(**self.kwargs)
            for _ in range(self.pop_size)
        ]

        # Initialize bests and evaluate initial fitness
        self.best = [p for p in self.population]  # local bests (copy of initial population)
        self.fitness_best = np.array([self.fitness_func(p) for p in self.population])  # fitness of each best

        # Initialize global best
        for i, particle in enumerate(self.population):
            fitness = self.fitness_func(particle)
            if self.global_fitness_best is not None:
                if fitness > self.global_fitness_best:
                    self.global_best = particle
                    self.global_fitness_best = fitness
            else:
                self.global_fitness_best = fitness
                self.global_best = particle

        # --- Main optimization loop ---
        for _ in range(n_iters):
            # Update local and global bests
            for i, particle in enumerate(self.population):
                fitness = self.fitness_func(particle)

                # Update local best if improved
                if self.fitness_best[i] is not None:
                    if fitness > self.fitness_best[i]:
                        self.fitness_best[i] = fitness
                        self.best[i] = particle

                # Update global best if improved
                if self.global_fitness_best is not None:
                    if fitness > self.global_fitness_best:
                        self.global_fitness_best = fitness
                        self.global_best = particle

            # Update velocity and position for each particle
            for i, particle in enumerate(self.population):
                # Select random neighbors (excluding self)
                indices = [j for j in range(self.pop_size) if j != i]
                neighbors = np.random.choice(indices, self.n_neighbors, replace=False)

                # Find the best neighbor based on fitness_best
                best_neighbor_index = max(neighbors, key=lambda j: self.fitness_best[j])
                neighbor_best = self.best[best_neighbor_index]

                # Update velocity using recombine with local, neighbor and global best
                particle.recombine(self.best[i], neighbor_best, self.global_best)
                # Update position based on velocity
                particle.mutate()
