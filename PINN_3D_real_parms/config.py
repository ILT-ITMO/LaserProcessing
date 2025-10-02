import physical_params as phys
from normalizer import QuartzNormalizer

# Создаем нормализатор
# normalizer = QuartzNormalizer.norm

# Безразмерные параметры для PINN
# NUM_GAUSSIANS = phys.NUM_GAUSSIANS
# GAUSSIAN_SPACING = phys.GAUSSIAN_SPACING
# SIGMA0 = dimensionless_params['sigma0_star']

# Параметры лазера (ДОБАВИТЬ MU_STAR)
# MU_STAR = dimensionless_params['mu_star']  # ← ДОБАВИТЬ ЭТУ СТРОКУ
# LASER_PULSE_DURATION = dimensionless_params['pulse_duration_star']
# LASER_PULSE_PERIOD = dimensionless_params['pulse_period_star']
# LASER_AMPLITUDE = dimensionless_params['laser_amplitude_star']
# LASER_SIGMA =

# Безразмерные границы области [0,1]
X_MIN = 0
X_MAX = 1
Y_MIN = 0
Y_MAX = 1
Z_MIN = 0
Z_MAX = 1
T_MAX = 1
# Коэффициент диффузии для PDE
# DIFF_COEF = dimensionless_params['alpha_star']