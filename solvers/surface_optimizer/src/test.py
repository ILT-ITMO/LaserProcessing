import numpy as np
import torch
from torch.utils import benchmark

from src.fire import FIRE
from src.lcp import LcpSolver
from main import solve


def benchmark_step(shape, device, use_threads=True):
    """
    Benchmarks the performance of a single step within the LcpSolver.
    
    This function measures the time taken to execute one step of the LcpSolver, 
    providing insights into its computational efficiency. It generates random input 
    data for the solver and utilizes benchmarking tools to assess performance 
    with and without multi-threading.
    
    Args:
        shape (tuple): The shape of the input array used to generate test data.
        device (str): The device (e.g., 'cpu', 'cuda') on which the solver operates.
        use_threads (bool, optional):  If True, utilizes multiple threads during the benchmark. 
                                       Defaults to True.
    
    Returns:
        None. Prints the benchmark results, including execution time, to the console.
    """
    b = np.random.random(shape)
    a = -np.random.random(np.fft.rfftn(b).shape)
    solver = LcpSolver(a, b, device=device)
    num_threads = torch.get_num_threads() if use_threads else 1
    t = benchmark.Timer(stmt='solver.step()', globals={'solver': solver},
                        num_threads=num_threads)
    print(t.blocked_autorange())


def benchmark_fft(size, device, use_threads=True):
    """
    Benchmarks the performance of the real-valued fast Fourier transform (rFFT) operation.
    
    Args:
        size: The size of the input tensor (tuple of integers).
        device: The device on which to perform the FFT (e.g., 'cpu', 'cuda').
        use_threads: Whether to utilize multiple threads during the FFT computation. Defaults to True.
    
    Returns:
        None. Prints the benchmark results, including execution time, to the console.
    """
    a = torch.randn(size, device=device, dtype=torch.float64)
    num_threads = torch.get_num_threads() if use_threads else 1
    t = benchmark.Timer(stmt='torch.fft.rfftn(a)', globals={'a': a},
                        num_threads=num_threads)
    print(t.blocked_autorange())


def rms(x):
    """
    Calculates the Root Mean Square (RMS) of a tensor.
    
    Args:
        x (torch.Tensor): The input tensor.
    
    Returns:
        float: The RMS value of the tensor.
    """
    return torch.sqrt(torch.mean(x ** 2)).item()


def test_fire():
    """
    Tests the FIRE optimization algorithm on the Rosenbrock function.
        
        This method sets the default PyTorch data type to float64, defines the
        Rosenbrock function's negative gradient, initializes a FIRE optimizer,
        and runs the optimization for a maximum of 3000 steps. It prints the
        RMS error at each step and breaks if the error falls below a threshold.
        The algorithm iteratively refines an initial guess towards a minimum of the Rosenbrock function.
        
        Args:
            None
        
        Returns:
            None
    """
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
    """
    Tests the solve function with various input parameters and validates the results.
    
    This method executes the `solve` function with a set of predefined parameters
    representing physical properties and boundary conditions. It then verifies
    that the returned `mean_gap` and `max_stress` values fall within acceptable
    ranges, ensuring the accuracy of the underlying physical model. Different
    parameter combinations are used to cover a range of scenarios and validate
    the robustness of the solution.
    
    Args:
        None
    
    Returns:
        None
    """
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
