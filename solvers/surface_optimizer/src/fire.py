from torch import zeros_like, norm, dot


class FIRE:
    """Implements Fast Inertial Relaxation Engine optimization algorithm:
    https://arxiv.org/abs/1908.02038
    """

    def __init__(self, neg_grad, initial_guess, dt_max):
        self.dt_max = dt_max
        self.dt = dt_max / 10
        self.n_pos = 0
        self.alpha = 0.25
        self.x = initial_guess
        self.v = zeros_like(initial_guess)
        self.neg_grad = neg_grad
        self.f = neg_grad(initial_guess)

    def step(self):
        f_norm = norm(self.f)
        if f_norm == 0.0:
            return

        if dot(self.v.ravel(), self.f.ravel()) > 0:
            self.n_pos += 1
            if self.n_pos > 20:
                self.dt = min(self.dt * 1.1, self.dt_max)
                self.alpha *= 0.99
        else:
            self.n_pos = 0
            self.v = zeros_like(self.v)
            self.dt = self.dt / 2
            self.alpha = 0.25

        self.v += 0.5 * self.dt * self.f
        f_dir = self.f / f_norm
        self.v = (1 - self.alpha) * self.v + self.alpha * norm(self.v) * f_dir
        self.x += self.dt * self.v
        self.f = self.neg_grad(self.x)
        self.v += 0.5 * self.dt * self.f

    def solution(self):
        return self.x
