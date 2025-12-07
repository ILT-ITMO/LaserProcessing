This code implements surface optimization for 2D elastic contact
problems. The goal is to find an optimal rigid surface profile that
minimizes a combined objective function of maximum interfacial stress and
contact stiffness when pressed against an elastic half-space.

Requirements:
- Python 3.7+
- NumPy
- PyTorch (for GFMD solver)
- Matplotlib (for plotting)
- cmaes library (pip install cmaes)

Solved problem:
- Optimizes Fourier coefficients defining a periodic surface profile
- Balances two competing objectives: minimizing peak contact stress vs
  maximizing contact stiffness
- Uses CMA-ES evolutionary algorithm with GFMD (Green's Function Molecular
  Dynamics) solver

How to run example:
1. Run "python example_optimize_surface.py"
2. The script performs multiple optimization trials with different random
   initializations
3. Results are saved to the specified directory (results/w=0.4 in the
   example)

Notes:
- Look inside example_optimize_surface.py to modify inputs before running
- Key parameters to adjust:
  - num_weights: Dimensionality of optimization (Fourier coefficients)
  - stress_weight: Balance between stress and stiffness objectives
  - num_trials: Independent optimizations from different starting points
  - num_points: Surface discretization resolution
- The optimization uses dimensionless units internally; see the conversion
  formulas in the file comments to obtain physical results

Output: Each trial creates plots of convergence history, final surface
profile, and contact stress distribution in the specified save directory.