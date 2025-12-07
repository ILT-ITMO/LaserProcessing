import numpy as np
import torch

from src.fire import FIRE


def rms(x):
    return torch.sqrt(torch.mean(x ** 2)).item()


class LcpSolver:
    def __init__(self, a, b, initial_guess=None, device='cpu'):
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
        if self.n_steps % 10 == 0:
            self.error_history[0].append(self.n_steps)
            self.error_history[1].append(self.error())

    def error(self):
        return rms(self.optimizer.f)

    def step(self):
        self.optimizer.step()
        self.update_error_history()
        self.n_steps += 1

    def solution(self):
        return self.optimizer.solution().cpu().numpy()
