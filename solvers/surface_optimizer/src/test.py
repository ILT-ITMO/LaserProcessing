import numpy as np
import torch
from torch.utils import benchmark

from src.fire import FIRE
from src.lcp import LcpSolver
from main import solve


def benchmark_step(shape, device, use_threads=True):
    b = np.random.random(shape)
    a = -np.random.random(np.fft.rfftn(b).shape)
    solver = LcpSolver(a, b, device=device)
    num_threads = torch.get_num_threads() if use_threads else 1
    t = benchmark.Timer(stmt='solver.step()', globals={'solver': solver},
                        num_threads=num_threads)
    print(t.blocked_autorange())


def benchmark_fft(size, device, use_threads=True):
    a = torch.randn(size, device=device, dtype=torch.float64)
    num_threads = torch.get_num_threads() if use_threads else 1
    t = benchmark.Timer(stmt='torch.fft.rfftn(a)', globals={'a': a},
                        num_threads=num_threads)
    print(t.blocked_autorange())


def rms(x):
    return torch.sqrt(torch.mean(x ** 2)).item()


def test_fire():
    torch.set_default_dtype(torch.float64)
    a = 1.0
    b = 100.0

    def rosenbrock_neg_grad(xy):
        (x, y) = xy
        return torch.tensor([2 * (a - x) + 4 * b * (y - x * x) * x,
                             2 * b * (x * x - y)])

    optimizer = FIRE(rosenbrock_neg_grad, torch.tensor([3.0, 4.0]),
                     dt_max=0.02)
    for i in range(3000):
        optimizer.step()
        d = rms(optimizer.solution() - torch.tensor([1.0, 1.0]))
        print(i + 1, f'{d :.20e}')
        if d < 1e-8:
            break


def test():
    nx = 2 ** 7
    ny = 2 ** 7
    x = np.linspace(-0.5, 0.5, nx, endpoint=False)
    y = np.linspace(-0.5, 0.5, ny, endpoint=False)
    length = 1.0
    (x, y) = np.meshgrid(x, y, indexing='xy')
    surface_1 = -0.5 * (x * x + y * y)

    result = solve(pressure=0.2, length=length,
                   surface_1=surface_1, E1=1e7, nu1=0.5,
                   E2=3.0, nu2=0.25, thickness_2=50.0)
    assert np.isclose(result['mean_gap'], 0.0245015)

    result = solve(pressure=0.2, length=length,
                   surface_1=surface_1, E1=(2 * 3.0), nu1=0.25,
                   E2=(2 * 3.0), nu2=0.25, thickness_2=50.0)

    assert np.isclose(result['max_stress'], 0.78204305)
    assert np.isclose(result['mean_gap'], 0.0245015)

    result = solve(pressure=0.2, length=length,
                   surface_1=surface_1, E1=1e7, nu1=0.25,
                   E2=3.0, nu2=0.25, thickness_2=0.25)

    assert np.isclose(result['mean_gap'], 0.0324626)
    assert np.isclose(result['max_stress'], 0.978424)
    result = solve(pressure=0.2, length=length,
                   surface_1=surface_1, E1=1e7, nu1=0.25,
                   E2=3.0, nu2=0.25, thickness_2=0.25)

    assert np.isclose(result['mean_gap'], 0.0324626)
    assert np.isclose(result['max_stress'], 0.978424)

    x = np.linspace(-0.5, 0.5, 128, endpoint=False)
    surface_1 = -0.5 * (x * x)
    result = solve(pressure=0.2, length=length,
                   surface_1=surface_1, E1=1e7, nu1=0.25,
                   E2=3.0, nu2=0.25, thickness_2=100.0)
    assert np.isclose(result['mean_gap'], 0.0132305)


if __name__ == '__main__':
    test()
