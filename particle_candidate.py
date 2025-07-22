import numpy as np
from softpy import FloatVectorCandidate

class ParticleCandidate(FloatVectorCandidate):
    '''
    Implementation of candidate solutions for Particle Swarm Optimization (PSO), 
    extending the FloatVectorCandidate with velocity and PSO-specific dynamics.

    Parameters
    ----------
    :param size: the number of components in the candidate solution
    :type size: int

    :param lower: lower bound of the search space (can be scalar or vector)
    :type lower: np.ndarray or float

    :param upper: upper bound of the search space (can be scalar or vector)
    :type upper: np.ndarray or float

    :param candidate: the initial position of the particle
    :type candidate: np.ndarray

    :param velocity: the initial velocity vector of the particle
    :type velocity: np.ndarray

    :param inertia: inertia weight (controls momentum from previous velocity)
    :type inertia: float, default=0.5

    :param wl: local best attraction weight
    :type wl: float, default=0.3

    :param wn: neighborhood best attraction weight
    :type wn: float, default=0.3

    :param wg: global best attraction weight
    :type wg: float, default=0.4
    '''
    def __init__(self, size, lower, upper, candidate, velocity,
                 inertia=0.5, wl=0.3, wn=0.3, wg=0.4):
        super().__init__(candidate)      # Initialize base class with candidate position
        self.size = size                 # Dimensionality of the solution
        self.lower = lower               # Lower bounds of the search space
        self.upper = upper               # Upper bounds of the search space
        self.velocity = velocity         # Velocity vector of the particle
        self.inertia = inertia           # Weight of current velocity
        self.wl = wl                     # Weight for local best attraction
        self.wn = wn                     # Weight for neighborhood best attraction
        self.wg = wg                     # Weight for global best attraction

    @classmethod
    def generate(cls, size, lower, upper,
                 inertia=0.5, wl=0.3, wn=0.3, wg=0.4):
        '''
        Factory method for generating a new particle with random position and velocity.

        Parameters
        ----------
        :param size: dimensionality of the problem
        :param lower: lower bounds
        :param upper: upper bounds
        :param inertia: inertia weight
        :param wl: local best weight
        :param wn: neighborhood best weight
        :param wg: global best weight

        Returns
        -------
        :return: an instance of ParticleCandidate
        :rtype: ParticleCandidate
        '''
        candidate = np.random.uniform(lower, upper)         # Random initial position
        vel_range = np.abs(upper - lower)                   # Velocity bounds based on position range
        velocity = np.random.uniform(-vel_range, vel_range) # Random initial velocity
        return cls(size, lower, upper, candidate, velocity, inertia, wl, wn, wg)

    def mutate(self):
        '''
        Updates the position of the particle using the current velocity.
        The position is then clipped to remain within the specified bounds.
        '''
        self.candidate = self.candidate + self.velocity
        self.candidate = np.clip(self.candidate, self.lower, self.upper)

    def recombine(self, local_best, neighborhood_best, global_best):
        '''
        Updates the velocity of the particle based on the PSO update rule, 
        combining contributions from inertia, local best, neighborhood best, and global best.

        Parameters
        ----------
        :param local_best: the particle's personal best position
        :param neighborhood_best: the best solution found among neighbors
        :param global_best: the best solution found globally
        '''
        rl = np.random.rand()  # Random weight for local component
        rn = np.random.rand()  # Random weight for neighborhood component
        rg = np.random.rand()  # Random weight for global component

        self.velocity = (
            self.inertia * self.velocity
            + rl * self.wl * (local_best.candidate - self.candidate)
            + rn * self.wn * (neighborhood_best.candidate - self.candidate)
            + rg * self.wg * (global_best.candidate - self.candidate)
        )
