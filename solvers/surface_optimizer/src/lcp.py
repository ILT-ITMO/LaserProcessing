import numpy as np
import torch

from src.fire import FIRE


def rms(x):
    """
    Calculates the Root Mean Square (RMS) of a tensor.
    
    Args:
      x (torch.Tensor): The input tensor.
    
    Returns:
      float: The RMS value of the tensor.
    """
    return torch.sqrt(torch.mean(x ** 2)).item()


class LcpSolver:
    """
    Solves the least common parent problem using optimization.
    
        Attributes:
            optimizer: The optimizer object used for solving the optimization problem.
            n_steps: The number of optimization steps taken.
            error_history: A tuple containing lists to store the error history during optimization.
    """

    def __init__(self, a, b, initial_guess=None, device='cpu'):
        """
        Initializes the optimization problem for solving a laser-induced heat transfer problem.
        
        Args:
            a: The coefficients representing the material properties and laser parameters for the Fourier transform. Must be non-positive.
            b: The constant term representing the background temperature or heat source.
            initial_guess: The initial temperature distribution as a tensor. If None, defaults to a tensor of zeros.
            device: The device to use for computations ('cpu' or 'gpu').
        
        Initializes the following object properties:
            self.optimizer: The FIRE optimizer object used for finding the temperature distribution.
            self.n_steps: An integer representing the number of optimization steps taken, initialized to 0.
            self.error_history: A tuple containing two lists to store the error history during optimization.
        
        Returns:
            None
        """
        assert np.all(a <= 0)

        a = torch.tensor(a, device=device)
        b = torch.tensor(b, device=device)

        min_period = 2 * np.pi * torch.max(-a).item() ** -0.5
        dt = min_period / 8

        if initial_guess is None:
            initial_guess = torch.zeros_like(b)
        else:
            initial_guess = torch.tensor(initial_guess, device=device)

        def neg_grad(x):
            torch.nn.functional.relu(x, inplace=True)
            f = torch.fft.irfftn(a * torch.fft.rfftn(x), b.shape) + b
            f[torch.logical_and(x == 0, f < 0)] = 0
            return f

        self.optimizer = FIRE(neg_grad, initial_guess, dt_max=dt)
        self.n_steps = 0

        self.error_history = ([], [])
        self.update_error_history()

    def update_error_history(self):
        """
        Updates the error history during the iterative solving process.
        
        Periodically records the step number and corresponding error value to track the convergence 
        and performance of the solution. This allows for monitoring the optimization process and 
        assessing the accuracy of the results over time.
        
        Args:
            self: The instance of the LcpSolver class.
        
        Returns:
            None
        """
        if self.n_steps % 10 == 0:
            self.error_history[0].append(self.n_steps)
            self.error_history[1].append(self.error())

    def error(self):
        """
        Calculates the root mean square (RMS) of the objective function values obtained during optimization.
        
        Args:
            self: The instance of the LcpSolver class.
        
        Returns:
            float: The RMS value of the objective function.  This represents a measure of the overall discrepancy between the model's predictions and the observed data during the optimization process.
        """
        return rms(self.optimizer.f)

    def step(self):
        """
        Performs a single optimization step to refine the solution.
        
        Updates the model's parameters to minimize the error, tracks the error evolution,
        and increments the step counter for monitoring progress.
        
        Args:
            self: The instance of the LcpSolver class.
        
        Returns:
            None
        """
        self.optimizer.step()
        self.update_error_history()
        self.n_steps += 1

    def solution(self):
        """
        Retrieves the optimal solution determined by the solver.
        
        Args:
            self: The instance of the LcpSolver class.
        
        Returns:
            numpy.ndarray: The optimal solution as a NumPy array, 
                           representing the result of the optimization process.
        """
        return self.optimizer.solution().cpu().numpy()
