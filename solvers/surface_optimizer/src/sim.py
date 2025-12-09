import pathlib
import time
from math import pi

import numpy as np
from numpy.fft import rfftfreq, fftfreq, irfftn, rfftn

from src.lcp import LcpSolver


def freq(n, d):
    """
    Computes the magnitude of the frequency spectrum for a 2D signal.
    
    Calculates the frequency components present in a signal based on its
    dimensions and sampling distance. This is useful for analyzing the
    spatial frequencies within the signal, which can reveal important
    information about its structure and characteristics.
    
    Args:
        n (int or list/tuple of int): Dimensions of the signal.  A single
            integer represents a 1D signal, while a list/tuple of two
            integers represents a 2D signal (rows, columns).
        d (float): Sample distance.
    
    Returns:
        ndarray: The magnitude of the frequency spectrum.  Represents the
            strength of each frequency component in the signal.
    """
    if len(n) == 1:
        return rfftfreq(n[0], d)
    xs = rfftfreq(n[1], d)
    ys = fftfreq(n[0], d)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    return np.sqrt(x * x + y * y)


def layer_factor(poisson, freq, thickness):
    """
    Calculates the layer factor to determine how a wave propagates through a material layer.
    
    This function computes the layer factor based on the material's properties (Poisson's ratio), 
    the wave's frequency, and the layer's thickness. It's designed to accurately model wave 
    behavior, particularly in scenarios involving laser-material interactions, by accounting 
    for potential numerical issues and specific layer conditions.
    
    Args:
        poisson (float): Poisson's ratio of the material, representing its resistance to deformation.
        freq (float): Frequency of the wave, determining its oscillatory rate.
        thickness (float): Thickness of the material layer, influencing wave propagation.
    
    Returns:
        numpy.ndarray: The calculated layer factor, a value between 0 and 1 that indicates 
                       the extent to which the wave is affected by the layer.
    """
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
    """
    Computes the Green's function in Fourier space for a layered medium, representing the material's response to harmonic excitation.
    
    Args:
        young (float): Young's modulus of the material.
        poisson (float): Poisson's ratio of the material.
        freq (float): Frequency of the excitation.
        thickness (float, optional): Thickness of the layer. Defaults to infinity.
    
    Returns:
        numpy.ndarray: The Green's function in Fourier space, representing the displacement field due to a point force. The first element is set to 0.0 to avoid singularities.
    """
    with np.errstate(divide='ignore'):
        result = 2 * (1 - poisson ** 2) / (young * 2 * pi * freq)
    result.flat[0] = 0.0
    return result * layer_factor(poisson, freq, thickness)


def gap_to_z(green_1, green_2, surface_1, gap):
    """
    Transforms the input surface based on Green's functions and a gap function, then calculates the difference between the transformed surface and the original gap.
    
    Args:
        green_1 (array_like): The first Green's function.
        green_2 (array_like): The second Green's function.
        surface_1 (array_like): The input surface data.
        gap (array_like): The gap function representing a spatial distribution.
    
    Returns:
        tuple: A tuple containing:
            - The difference between the transformed surface and the gap (array_like).
            - The transformed surface (array_like).
    
    The method leverages Green's functions to modify the input surface, effectively modeling a physical response to an external influence represented by the gap function. This transformation is performed in the Fourier domain for efficiency, and the final result represents a refined surface accounting for the interaction between the initial surface, the gap, and the properties defined by the Green's functions.
    """
    with np.errstate(invalid='ignore'):
        a = 1 / (1 + green_1 / green_2)
    a.flat[0] = 0.0
    z_2 = irfftn(a * rfftn(surface_1 + gap), gap.shape, range(gap.ndim))
    return z_2 - gap, z_2


def inv(x):
    """
    Computes the inverse of an array, handling potential division by zero errors that can occur during data processing.
    
    Args:
        x (numpy.ndarray): The input array.
    
    Returns:
        numpy.ndarray: An array containing the inverse of the input array,
        with division by zero resulting in 0 to maintain numerical stability.
    """
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
    """
    Solves the contact problem between two elastic bodies under external pressure.
    
    This method determines the deformation and stress distribution resulting from the interaction
    of two elastic bodies subjected to a given pressure, leveraging Fourier transforms and a 
    Linear Complementarity Problem (LCP) solver. The solution provides insights into the 
    contact mechanics of the system.
    
    Args:
        pressure: The external pressure applied to the bodies.
        length: The length of the bodies.
        surface_1: The shape of the first body's surface.
        E1: Young's modulus of the first body.
        nu1: Poisson's ratio of the first body.
        E2: Young's modulus of the second body.
        nu2: Poisson's ratio of the second body.
        thickness_2: The thickness of the second body (default is infinity).
        maxiter: The maximum number of iterations for the LCP solver (default is 1000000).
        tol: The tolerance for the LCP solver (default is 1e-8).
        shear_str: The shear strength (default is None).
        use_cuda: Whether to use CUDA for the LCP solver (default is False).
        initial_guess: The initial guess for the gap (default is None).
        log_output: Whether to print log output (default is False).
        dump_dir: The directory to dump the results to (default is None).
    
    Returns:
        A dictionary containing the results of the solution, including the mean gap,
        relative contact area, maximum stress, stress distribution, deformed surfaces,
        gap, and number of iterations. If shear strength is provided, the friction
        coefficient is also included. If a dump directory is provided, the results
        are saved to files in that directory.
    """
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
