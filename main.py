import numpy as np
from pso_optimizer import ParticleSwarmOptimizer


def sphere_fitness(particle) -> float:
    '''
    Fitness function for the Sphere benchmark.

    Parameters
    ----------
    :param particle: the candidate particle whose fitness is being evaluated
    :type particle: ParticleCandidate

    Returns
    -------
    :return: negative squared Euclidean norm of the particle vector
    :rtype: float

    Notes
    -----
    The Sphere function is defined as: f(x) = sum(x_i^2)
    Since PSO maximizes by default, we return -f(x).
    Global minimum is at x=0 with f(x)=0.
    '''
    return -np.sum(particle.candidate ** 2)


if __name__ == "__main__":
    '''
    Example usage of ParticleSwarmOptimizer on the Sphere function.
    Runs a PSO optimization in a 5D search space bounded in [-5, 5].

    Prints the best solution and corresponding fitness found.
    '''

    # --- Problem definition ---
    size = 5  # number of dimensions
    lower = np.array([-5.0] * size)  # lower bounds for each dimension
    upper = np.array([5.0] * size)   # upper bounds for each dimension

    # --- Initialize PSO optimizer ---
    pso = ParticleSwarmOptimizer(
        fitness_func=sphere_fitness,
        pop_size=30,
        n_neighbors=5,
        size=size,
        lower=lower,
        upper=upper
    )

    # --- Run optimization ---
    pso.fit(n_iters=50)

    # --- Print results ---
    print("Best fitness found:", pso.global_fitness_best)
    print("Best solution found:", str(pso.global_best))
