"""All settings of a discovery run in one place.

Fields in the first block can be changed from the command line (see train.py);
the remaining fields are the fixed settings used for the paper's results.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Command-line options -------------------------------------------------
    seed: int = 13                    # torch seed, reset at the start of every (model, noise) run
    num_epochs_adam: int = 20000      # Stage 1 epochs
    ado_iterations: int = 10          # Stage 2 cycles (L-BFGS + sparse regression)
    lbfgs_max_iter: int = 2000        # L-BFGS iterations per cycle
    num_initial_guesses: int = 3000   # sparse regression: rounds of 3 random guesses (+ current theta)
    show_plots: bool = False          # show intermediate diagnostic figures

    # ---- Problem / data -------------------------------------------------------
    data_subdir: str = 'Circular_Hole_One_Quarter_Model/Hole_Radius_50_Tension_100_Plane_Strain_F33_-0.5'
    hole_radius: float = 50.0
    f33: float = -0.5                 # prescribed F33 at the last increment is 1 + f33
    num_increments: int = 10
    measurement_angles_deg: tuple = (0, 22.5, 45, 67.5, 90)  # measurement points on the hole edge
    trial: int = 1                    # only used in the output file names

    # ---- Network --------------------------------------------------------------
    width: int = 32

    # ---- Loss -----------------------------------------------------------------
    data_loss_weight: float = 0.5     # weight of the measurement-point displacement loss
    near_hole_margin: float = 1.0     # free DOFs within hole_radius + margin form the near-hole residual

    # ---- Stage 1: Adam --------------------------------------------------------
    lr_adam: float = 5e-4             # network and coefficients
    grad_clip: float = 1.0            # max gradient norm of the network
    theta_min: float = 1e-4           # coefficients below this are set to 0 after each step

    # ---- Stage 2: L-BFGS (network only) ---------------------------------------
    lbfgs_lr: float = 1
    lbfgs_tolerance_grad: float = 1e-15
    lbfgs_tolerance_change: float = 1e-15
    lbfgs_history_size: int = 100

    # ---- Stage 2: Lp sparse regression (coefficients only) --------------------
    lp_p: float = 1 / 4
    lp_threshold: float = 1e-3
    # lambda = factor * sum(FF^2), per benchmark model
    lp_lambda_factors: dict = field(default_factory=lambda: {
        'NH2': 3e-3, 'NH4': 3e-3, 'MR2': 3e-3, 'MR4': 3e-3, 'YH': 3e-3, 'IH': 3e-3,
        'HW': 1e-3,
        'HN': 2e-3,
    })
    lp_lambda_factor_default: float = 10e-3

    # ---- Logging --------------------------------------------------------------
    print_every: int = 100
    plot_every: int = 500             # loss / coefficient plots (with show_plots)
    field_plot_every: int = 1000      # field plots (with show_plots)

    def lambda_factor(self, model_symbol):
        return self.lp_lambda_factors.get(model_symbol, self.lp_lambda_factor_default)
