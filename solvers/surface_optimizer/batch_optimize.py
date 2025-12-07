from optimize_surface import optimize_trials

n = 50
for i in range(n - 1):
    w = (i + 1) / n
    num_trials = 1
    optimize_trials(num_weights=5, num_trials=num_trials, mean_rng_range=10.0,
                    save_dir=f'optimization_results/w={w}', stress_weight=w,
                    seed=i * num_trials, even_func=True)
