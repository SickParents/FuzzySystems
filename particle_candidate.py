

import numpy as np
from softpy import FloatVectorCandidate


class ParticleCandidate(FloatVectorCandidate):
    def __init__(self, size, lower, upper, candidate, velocity,
                 inertia=0.5, wl=0.3, wn=0.3, wg=0.4):
        super().__init__(candidate)
        self.size = size
        self.lower = lower
        self.upper = upper
        self.velocity = velocity
        self.inertia = inertia
        self.wl = wl
        self.wn = wn
        self.wg = wg

    @classmethod
    def generate(cls, size, lower, upper,
                 inertia=0.5, wl=0.3, wn=0.3, wg=0.4):
        candidate = np.random.uniform(lower, upper)
        vel_range = np.abs(upper - lower)
        velocity = np.random.uniform(-vel_range, vel_range)
        return cls(size, lower, upper, candidate, velocity, inertia, wl, wn, wg)

    def mutate(self):
        self.candidate = self.candidate + self.velocity
        self.candidate = np.clip(self.candidate, self.lower, self.upper)

    def recombine(self, local_best, neighborhood_best, global_best):
        rl = np.random.rand()
        rn = np.random.rand()
        rg = np.random.rand()

        self.velocity = (
            self.inertia * self.velocity
            + rl * self.wl * (local_best.candidate - self.candidate)
            + rn * self.wn * (neighborhood_best.candidate - self.candidate)
            + rg * self.wg * (global_best.candidate - self.candidate)
        )
