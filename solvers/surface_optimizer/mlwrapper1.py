import numpy as np

from src.sim import solve


def mlwrapper1(
        pressure,
        length,
        surface_1,
        E2, nu2,
        thickness_2=np.inf,
        maxiter=1000000,
        tol=1e-8,
        use_cuda=False,
        initial_guess=None,
        log_output=False,
):
    """
    Simulate 2D contact between a rigid body (body 1) and an elastic layer
        (body 2) under given pressure to determine the resulting interfacial stress and deformation.
    
        Parameters:
            pressure (float): Pressure of contact (Pa)
            length (float): Unit cell size in one dimension (m)
            surface_1 (ndarray): Surface topography unit cell height data (the
                mesh) of body 1 (m)
            E2 (float): Young modulus of body 2 (Pa)
            nu2 (float): Poisson ratio of body 2
            thickness_2 (float, optional): Thickness of body 2 (m). Body 2 is
                indented by body 1 on one side and bonded to a rigid flat surface
                on the other side. Defaults to np.inf (semi-infinite medium)
            maxiter (int, optional): Maximum number of iterations
            tol (float, optional): Convergence tolerance threshold
            use_cuda (bool, optional): Use GPU acceleration
            initial_guess (ndarray, optional): Initial guess for the gap array
                between the surfaces (m)
            log_output (bool, optional): Print solver progress
    
        Returns:
            dict: Dictionary containing:
                - max_stress (float): Maximum interfacial stress (Pa)
                - contact_stiffness (float): Contact stiffness (Pa/m)
                - stress (ndarray): Interfacial stress array (Pa)
                - body_2_surface (ndarray): Deformed surface profile of body 2 (m)
                - gap (ndarray): Final gap array between the surfaces (m). (Can be
                    used as initial_guess in follow-up simulations)
    """
    E1 = 1e16 * E2
    nu1 = 0.5
    result = solve(
        pressure,
        length,
        surface_1,
        E1, nu1,
        E2, nu2,
        thickness_2=thickness_2,
        maxiter=maxiter,
        tol=tol,
        use_cuda=use_cuda,
        initial_guess=initial_guess,
        log_output=log_output,
    )
    z2 = result['body_2_surface']
    result['contact_stiffness'] = pressure / (np.max(z2) - np.min(z2))
    result['length'] = length
    return result


if __name__ == "__main__":
    # Example usage
    x = np.linspace(-0.5, 0.5, 128, endpoint=False)
    surface_1 = -0.5 * (x * x)  # Parabolic surface
    result = mlwrapper1(
        pressure=0.2, length=1.0, surface_1=surface_1, E2=3.0, nu2=0.25
    )
    print(result['contact_stiffness'])
    print(result['max_stress'])
