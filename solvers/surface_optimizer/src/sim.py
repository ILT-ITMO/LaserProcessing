import pathlib
import time
from math import pi

import numpy as np
from numpy.fft import rfftfreq, fftfreq, irfftn, rfftn

from src.lcp import LcpSolver


def freq(n, d):
    if len(n) == 1:
        return rfftfreq(n[0], d)
    xs = rfftfreq(n[1], d)
    ys = fftfreq(n[0], d)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    return np.sqrt(x * x + y * y)


def layer_factor(poisson, freq, thickness):
    # https://doi.org/10.1093/qjmam/4.1.94
    if thickness == np.inf:
        return 1.0
    a = 2 * pi * freq * thickness
    b = 3 - 4 * poisson
    with np.errstate(all='ignore'):
        c = b * np.sinh(2 * a) - 2 * a
        c /= b * np.cosh(2 * a) + 2 * a * a + b + 2 * (1 - 2 * poisson) ** 2
    c[a > 40] = 1
    c.flat[0] = 1
    return c


def green_func_fourier(young, poisson, freq, thickness=np.inf):
    with np.errstate(divide='ignore'):
        result = 2 * (1 - poisson ** 2) / (young * 2 * pi * freq)
    result.flat[0] = 0.0
    return result * layer_factor(poisson, freq, thickness)


def gap_to_z(green_1, green_2, surface_1, gap):
    with np.errstate(invalid='ignore'):
        a = 1 / (1 + green_1 / green_2)
    a.flat[0] = 0.0
    z_2 = irfftn(a * rfftn(surface_1 + gap), gap.shape, range(gap.ndim))
    return z_2 - gap, z_2


def inv(x):
    with np.errstate(divide='ignore'):
        x = 1 / x
    return np.nan_to_num(x, 0, 0, 0)


def solve(
        pressure,
        length,
        surface_1,
        E1, nu1,
        E2, nu2,
        thickness_2=np.inf,
        maxiter=1000000,
        tol=1e-8,
        shear_str=None,
        use_cuda=False,
        initial_guess=None,
        log_output=False,
        dump_dir=None,
):
    shape = surface_1.shape
    f = freq(shape, length / shape[-1])
    ft_green_1 = green_func_fourier(E1, nu1, f)
    ft_green_2 = green_func_fourier(E2, nu2, f, thickness_2)
    ft_green_eff = ft_green_1 + ft_green_2

    a = -inv(ft_green_eff)
    b = irfftn(a * rfftn(surface_1), shape, range(len(shape))) - pressure

    (a, b) = (a / -a.flat[1], b / -a.flat[1])
    if initial_guess is None:
        initial_guess = np.max(surface_1) - surface_1
    device = 'cuda' if use_cuda else 'cpu'
    lcp = LcpSolver(a, b, initial_guess=initial_guess, device=device)

    t0 = time.perf_counter_ns()
    success = False
    start_str = f'P={pressure:.4},'

    for i in range(maxiter):
        if i % 20 == 0:
            error = lcp.error()
            if log_output:
                print(start_str, f'step={i}, |F|={error:.2e}')
            if error < tol:
                success = True
                break
        lcp.step()

    if log_output:
        if lcp.n_steps != 0:
            total = time.perf_counter_ns() - t0
            step = total / lcp.n_steps
            print(start_str,
                  f'{total / 1e9:.2} sec total, '
                  f'{int(step / 1e3)} usec per step')
        print(start_str, 'success' if success else 'failure')

    gap = lcp.solution()
    z_1, z_2 = gap_to_z(ft_green_1, ft_green_2, surface_1, gap)

    stress = pressure + irfftn(inv(ft_green_2) * rfftn(z_2),
                               shape, range(len(shape)))

    result = {
        'mean_gap': np.mean(gap),
        'rel_contact_area': np.mean(gap < 1e-14),
        'max_stress': np.max(stress),
        'stress': stress,
        'body_1_surface': z_1,
        'body_2_surface': z_2,
        'gap': gap,
        'n_steps': lcp.n_steps,
    }
    if shear_str is not None:
        result['friction_coeff'] = (
                result['rel_contact_area'] * shear_str / pressure)

    if dump_dir is not None:
        path = pathlib.Path(dump_dir)
        path.mkdir(exist_ok=True, parents=True)
        x = np.linspace(0, length, shape[0], endpoint=False)
        if len(shape) == 2:
            (x, y) = np.meshgrid(x, x)
            np.savetxt(path / 'y.txt', y)
        np.savetxt(path / 'x.txt', x)
        np.savetxt(path / 'z1.txt', result['body_1_surface'])
        np.savetxt(path / 'z2.txt', result['body_2_surface'])
        np.savetxt(path / 'stress.txt', result['stress'])

    return result
