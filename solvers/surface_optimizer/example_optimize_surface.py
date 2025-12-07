from optimize_surface import optimize_trials

# The optimization uses normalized dimensionless units internally (p, L, E_star
# are set to 1). To obtain physical results for specific material and loading
# conditions, multiply the dimensionless outputs as follows:
#
# Let:
#     E = Young modulus (Pa)
#     nu = Poisson ratio
#     L = Domain length (m)
#     p = Applied pressure (Pa)
#     E_star = E / (1 - nu^2)   (Effective elastic modulus)
#
# Conversion factors:
#     Surface height (z):      Multiply by  (L * p / E_star)
#     Horizontal position (x): Multiply by  L
#     Contact stress (σ):      Multiply by  p
#     Contact stiffness (K):   Multiply by  (E_star / L)

optimize_trials(
    # Number of Fourier coefficients to optimize.
    # Determines the dimensionality of the optimization problem.
    num_weights=5,

    # Number of independent optimization trials to run.
    # Each trial starts from a different random initialization.
    num_trials=3,

    # Range for random initialization of the CMA-ES mean vector.
    # The first coefficient is initialized uniformly in [0, mean_rng_range].
    # Remaining coefficients are initialized uniformly in
    # [-mean_rng_range, mean_rng_range].
    mean_rng_range=10.0,

    # Directory path where trial results will be saved.
    # The directory will be created if it doesn't exist, and any existing
    # content will be removed.
    save_dir='results/w=0.4',

    # Base random seed for reproducibility.
    # Each trial i uses seed = base_seed + i.
    seed=0,

    # Weight for stress component in the combined objective function.
    # - Range: (0, 1)
    # - objective = w * normalized_stress + (1 - w) * normalized_stiffness
    stress_weight=0.4,

    # Initial step size (standard deviation) for CMA-ES optimizer.
    # Controls the exploration radius in parameter space.
    sigma=1.0,

    # Number of discretization points for the surface profiles.
    num_points=512,

    # Minimum improvement of the objective function required to continue
    # optimization. If improvement < threshold for max_no_improvement
    # generations, early stopping is triggered.
    improvement_threshold=1e-5,

    # Number of consecutive generations without significant improvement
    # before triggering early stopping.
    max_no_improvement=50,

    # GFMD solver tolerance
    tol=1e-8,

    # Maximum number of generations (iterations) for CMA-ES.
    num_generations=4000,

    # Whether to constrain the surface to an even function.
    # - True: Uses only cosine terms, symmetric surface
    # - False: Uses both cosine and sine terms, allows surface asymmetry
    # For even_func=False, num_weights must be odd.
    even_func=True
)
