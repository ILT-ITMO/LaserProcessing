from torch import zeros_like, norm, dot


class FIRE:
    """
    Implements Fast Inertial Relaxation Engine optimization algorithm:
        https://arxiv.org/abs/1908.02038
    """


    def __init__(self, neg_grad, initial_guess, dt_max):
        """
        Initializes an instance of the class.
        
        Args:
            neg_grad (callable): A function that returns the negative gradient of the potential.
            initial_guess (ndarray): The initial guess for the optimization.
            dt_max (float): The maximum step size.
        
        Initializes the following object properties:
            dt_max (float): The maximum step size.
            dt (float): The current step size, initialized to dt_max / 10.
            n_pos (int): A counter for positive definite steps, initialized to 0.
            alpha (float): A parameter controlling step size adjustment, initialized to 0.25.
            x (ndarray): The current position, initialized to the initial guess.
            v (ndarray): The current velocity, initialized to a zero array with the same shape as the initial guess.
            neg_grad (callable): The negative gradient function.
            f (ndarray): The force, initialized to the negative gradient at the initial guess.
        
        Returns:
            None
        """
        self.dt_max = dt_max
        self.dt = dt_max / 10
        self.n_pos = 0
        self.alpha = 0.25
        self.x = initial_guess
        self.v = zeros_like(initial_guess)
        self.neg_grad = neg_grad
        self.f = neg_grad(initial_guess)

    def step(self):
        """
        Steps the optimization process forward.
        
        Updates the position, velocity, and step size based on the current gradient to efficiently navigate towards a minimum. The method dynamically adjusts parameters to accelerate convergence when moving in a consistent direction and dampens oscillations when encountering resistance.
        
        Args:
            self: The instance of the FIRE class.
        
        Returns:
            None
        
        Class Fields Initialized:
            n_pos: Counter for consecutive positive dot products of velocity and gradient, influencing step size.
            dt: The step size, adjusted based on optimization progress.
            alpha: The momentum factor, controlling the influence of past velocities.
            v: The velocity vector, updated with gradient information.
            x: The current position vector, representing the optimized parameters.
            f: The current gradient vector, indicating the direction of steepest ascent.
        """
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
        """
        Returns the x-coordinate of the current point.
        
        Args:
            self: The instance of the FIRE class.
        
        Returns:
            int: The x-coordinate value.
        """
        return self.x
