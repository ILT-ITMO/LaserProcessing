import csv
import json
import os
import pathlib
import shutil

import numpy as np
from cmaes import CMA
from matplotlib import pyplot as plt

from mlwrapper1 import mlwrapper1


def fourier_series(x, period, a_coeffs, b_coeffs):
    n_cos = np.arange(len(a_coeffs))
    cos_terms = np.cos(2 * np.pi * n_cos[:, None] * x[None, :] / period)

    n_sin = np.arange(1, len(b_coeffs) + 1)
    sin_terms = np.sin(2 * np.pi * n_sin[:, None] * x[None, :] / period)

    return np.dot(a_coeffs, cos_terms) + np.dot(b_coeffs, sin_terms)


def plot_objective_history(path):
    data = np.genfromtxt(path / 'best_per_epoch.csv', names=True,
                         delimiter=',')

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(data['epoch'], data['objective'])

    ax.set_xlabel('Generation')
    ax.set_ylabel('Objective value')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    plt.savefig(path / 'optimization_convergence.png', dpi=300,
                bbox_inches='tight')
    plt.savefig(path / 'optimization_convergence.pdf',
                bbox_inches='tight')


def plot_surfaces_and_stress(path):
    x, z_1, z_2, stress = np.loadtxt(path / 'stress.txt')
    z_1[z_1 != z_2] = np.nan

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1],
                                        'hspace': 0.1})

    ax[0].plot(x, (z_2 - z_2.min()), '--', label='Elastic body surface')
    ax[0].plot(x, (z_1 - z_2.min()),
               label='Optimized surface (active part)', linewidth=2.5)
    ax[0].set_ylabel('Height')
    ax[0].grid(True, alpha=0.3, linestyle='--')
    ax[0].legend(prop={'size': 7})

    ax[1].plot(x, stress, color='#C44536', linewidth=2.5)
    ax[1].set_xlabel('Position')
    ax[1].set_ylabel('Stress')
    ax[1].grid(True, alpha=0.3, linestyle='--')

    plt.subplots_adjust(hspace=0.1)

    plt.savefig(path / 'surfaces_and_stress.png', dpi=300, bbox_inches='tight')
    plt.savefig(path / 'surfaces_and_stress.pdf', bbox_inches='tight')


def _save_best(epoch: int, best_value: float, stress: float, stiffness: float,
               best_weights: np.ndarray, path: str):
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            header = ["epoch", "objective", "stress", "stiffness"]
            header += [f"w{i}" for i in range(len(best_weights))]
            w.writerow(header)

        row = ([epoch, float(best_value), stress, stiffness]
               + [float(x) for x in best_weights])
        w.writerow(row)


def random_mean(size, high, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.uniform(0, high, 1),
                           rng.uniform(-high, high, size - 1)))


def optimize_trials(num_weights, num_trials, mean_rng_range, save_dir, seed=0,
                    **kwargs):
    path = pathlib.Path(save_dir)

    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)

    config = {
        'num_weights': num_weights,
        'num_trials': num_trials,
        'mean_rng_range': mean_rng_range,
        'seed': seed,
        'kwargs': kwargs,
    }
    with open(path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    objective_values = []
    for i in range(num_trials):
        print(f"Running trial {i}/{num_trials}")
        mean = random_mean(num_weights, mean_rng_range, seed + i)
        result = optimize_surface(mean=mean, save_dir=path / f'{i}',
                                  seed=seed + i, **kwargs)
        objective_values.append(result)

    sorted_results = sorted(enumerate(objective_values), key=lambda x: x[1])
    with open(path / 'results_sorted.txt', 'w') as f:
        f.write(f"{'#':3s}  {'objective':>10s}\n")
        for i, objective in sorted_results:
            f.write(f'{i:3d}  {objective:10.6f}\n')


def make_surface(weights, length, num_points, even_func=False):
    x = np.linspace(-length / 2, length / 2, num_points, endpoint=False)
    if even_func:
        a_coeffs = np.concatenate(([0.0], weights))
        b_coeffs = []
    else:
        assert len(weights) % 2 == 1
        half = (len(weights) - 1) // 2 + 1
        a_coeffs = np.concatenate(([0.0], weights[:half]))
        b_coeffs = np.concatenate(([0.0], weights[half:]))
    return fourier_series(x, length, a_coeffs, b_coeffs)


def objective(stress_weight, num_points, weights, pressure, length, young,
              poisson,
              tol, even_func):
    surface = make_surface(weights, length, num_points, even_func)
    mlwrapper1_result = mlwrapper1(pressure=pressure, length=length,
                                   surface_1=surface, E2=young,
                                   nu2=poisson, tol=tol)
    stiffness = mlwrapper1_result['contact_stiffness']
    stress_ = mlwrapper1_result['max_stress'] / pressure
    stiffness_ = stiffness / (young / (1 - poisson ** 2) / length)
    w = stress_weight
    value = w * stress_ + (1 - w) * stiffness_
    return value, mlwrapper1_result, stiffness_, stress_


def optimize_surface(
        mean,
        sigma=1.0,
        save_dir='output',
        seed=0,
        num_generations=4000,
        pressure=1.,
        length=1.,
        young=1.,
        poisson=0.,
        num_points=512,
        stress_weight=0.5,
        tol=1e-6,
        even_func=False,
        improvement_threshold=1e-5,
        max_no_improvement=50,
):
    path = pathlib.Path(save_dir)
    path.mkdir(exist_ok=True, parents=True)

    with open(path / 'inputs.json', 'w', encoding='utf-8') as f:
        json.dump({
            'stress_weight': stress_weight,
            'num_weights': len(mean),
            'num_generations': num_generations,
            'cma_seed': seed,
            'cma_sigma': sigma,
            'num_points': num_points,
            'pressure': pressure,
            'length': length,
            'young_modulus': young,
            'poisson_ratio': poisson,
            'gfmd_tolerance': tol,
            'even_func': even_func,
            'sigma': sigma,
        }, f, indent=2)

    bounds = []
    for i in range(len(mean)):
        if i == 0:
            bounds.append([0.0, np.inf])
        else:
            bounds.append([-np.inf, np.inf])
    bounds = np.array(bounds)

    optimizer = CMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)

    objective_history = []

    try:
        for generation in range(num_generations):
            solutions = []
            for _ in range(optimizer.population_size):
                weights = optimizer.ask()

                value, mlwrapper1_result, stiffness_, stress_, = objective(
                    stress_weight, num_points, weights, pressure, length, young,
                    poisson, tol, even_func)

                solutions.append((weights, value, mlwrapper1_result,
                                  stress_, stiffness_))
                print(f"#{generation}\t"
                      f"objective={value:.6f}\t"
                      f"σ_max={stress_:.4f}\t"
                      f"K={stiffness_:.4f}\t"
                      f"(weights[:3]={weights[:3]})")

            best_x, best_value, mlwrapper1_result, stress_, stiffness_ = (
                min(solutions, key=lambda item: item[1]))

            _save_best(generation, best_value, stress_, stiffness_,
                       np.asarray(best_x, dtype=float),
                       path / "best_per_epoch.csv")

            optimizer.tell(solutions)

            objective_history.append(best_value)

            idx_compare = len(objective_history) - 1 - max_no_improvement
            if idx_compare >= 0:
                improvement = abs(objective_history[idx_compare]
                                  - objective_history[-1])
                if improvement < improvement_threshold:
                    print(f"Early stopping at generation {generation} - no "
                          f"significant improvement in {max_no_improvement} "
                          f"generations")
                    break
    except KeyboardInterrupt:
        pass

    best_x, best_value, mlwrapper1_result, _, _ = min(solutions,
                                                      key=lambda item: item[1])

    z_1 = mlwrapper1_result['body_1_surface']
    z_2 = mlwrapper1_result['body_2_surface']
    stress = mlwrapper1_result['stress']

    is_active = z_1 == z_2
    z = z_1.copy()
    z[~is_active] = z_2[~is_active]
    np.savetxt(path / 'surface.txt', np.array([z, is_active]).T, fmt='%s %d',
               header='z is_active')

    z_1 = np.array([*z_1, z_1[0]])
    z_2 = np.array([*z_2, z_2[0]])
    stress = np.array([*stress, stress[0]])

    x = np.linspace(0.0, mlwrapper1_result['length'], len(z_1), endpoint=True)

    np.savetxt(path / 'stress.txt', np.array([x, z_1, z_2, stress]))
    plot_surfaces_and_stress(path)
    plot_objective_history(path)
    return best_value
