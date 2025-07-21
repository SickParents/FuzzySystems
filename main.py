import numpy as np
from pso_optimizer import ParticleSwarmOptimizer
from particle_candidate import ParticleCandidate


# Funzione da massimizzare: Sphere (minimo globale = 0, massimo = 0)
def sphere_fitness(particle):
    return -np.sum(particle.candidate ** 2)

# Parametri del problema
size = 5
lower = np.array([-5.0] * size)
upper = np.array([5.0] * size)

# Inizializza ottimizzatore
pso = ParticleSwarmOptimizer(
    fitness_func=sphere_fitness,
    pop_size=30,
    n_neighbors=5,
    size=size,
    lower=lower,
    upper=upper
)

# Esegui ottimizzazione
pso.fit(n_iters=50)

# Stampa i risultati
print("Miglior fitness trovata:", pso.global_fitness_best)
print("Miglior soluzione trovata:", pso.global_best)
