"""PINN-based discovery of hyperelastic strain-energy models (two-stage training).

Accompanies "Physics-informed neural network-based discovery of hyperelastic
constitutive models from extremely scarce data" (CMAME, 2025).

A PINN predicts the displacement field u(X, t) of a plate with a circular hole
(quarter model, plane strain with prescribed F33) at 10 load increments. The
deformation gradient is obtained from the PINN displacements with linear
triangle shape functions, and the weak-form force balance A(u) theta = F is
built from a library of 57 candidate strain-energy terms. Only boundary
displacements, 5 points on the hole edge and the edge reaction forces are used.

Training:
  Stage 1  Adam jointly on the network and the 57 coefficients theta
           (theta < 1e-4 is set to 0 after every step).
  Stage 2  Alternation of (a) L-BFGS on the network only (theta fixed) and
           (b) Lp (p = 1/4) thresholded sparse regression on theta.

Outputs (in --output-dir, <tag> = <model>_Noise_<noise>_Trial_1):
  Params_<tag>.jpg, Loss_<tag>.jpg, Loss_<tag>.npy,
  Consitutive_parameter_<tag>.npy (history of theta),
  AI_Model_<tag> (torch checkpoint with 'model_state_dict' and
  'constitutive_params_state_dict')

Usage (a GPU is used if available):
  python PINN/train.py                         # all models, all noise levels
  python PINN/train.py --model HN --noise 0.01
All fixed settings are in utils/config.py.
"""
import argparse
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.config import Config
from utils.data import add_noise, load_fem_data, total_reaction_forces
from utils.library import MODEL_SYMBOLS, NUM_TERMS, true_theta
from utils.loss import build_loss_context, compute_diagnostics, compute_losses, new_loss_history
from utils.mesh import load_mesh, select_measurement_nodes
from utils.network import ConstitutiveParams, PINN
from utils.plotting import (plot_loss_history, plot_mesh, plot_params, plot_reaction_force_noise,
                            save_loss_history_plot, save_params_plot, show_field_diagnostics)
from utils.sparse_regression import main_Lp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05]


def set_seed(seed):
    """Seed torch (CPU and CUDA) and make cuDNN deterministic. NumPy is not seeded."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_stage1(model, constitutive_params, ctx, loss_history, theta_true, mesh, data, cfg, device):
    """Stage 1: Adam on the network (PINN loss) and theta (unscaled weak-form loss).

    Returns the start time (used for the elapsed time printed in Stage 2).
    """
    optimizer_model = torch.optim.Adam(model.parameters(), lr=cfg.lr_adam)
    optimizer_constitutive = torch.optim.Adam(constitutive_params.parameters(), lr=cfg.lr_adam)

    start_time = time.time()
    for epoch in range(1, cfg.num_epochs_adam + 1):
        model.train()
        out = compute_losses(model, constitutive_params, ctx, loss_history)

        optimizer_model.zero_grad()
        optimizer_constitutive.zero_grad()

        model_grads = torch.autograd.grad(out.pinn_loss, model.parameters(), create_graph=True, allow_unused=True)
        for param, grad in zip(model.parameters(), model_grads):
            if grad is not None:
                param.grad = grad.clone()

        constitutive_grads = torch.autograd.grad(out.constitutive_loss, constitutive_params.parameters(),
                                                 create_graph=True, allow_unused=True)
        for param, grad in zip(constitutive_params.parameters(), constitutive_grads):
            if grad is not None:
                param.grad = grad.clone()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer_model.step()
        optimizer_constitutive.step()

        # Keep the coefficients non-negative (small values are set to zero)
        new_params = constitutive_params.params.clone().detach()
        new_params[new_params < cfg.theta_min] = torch.tensor(0.0, device=device)
        constitutive_params.update_constitutive_params(new_params)
        constitutive_params.save_param_history()

        if epoch % cfg.print_every == 0:
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"Adam solver : Epoch {epoch} \t PINN Loss = {out.pinn_loss:.4e} \t "
                  f"Consitutive Loss = {out.constitutive_loss:.4e} \t Elapsed Time = {elapsed_time_str}")

        if epoch % cfg.plot_every == 0:
            if cfg.show_plots:
                plot_loss_history(loss_history)
                plot_params(constitutive_params.params, theta_true, epoch)
            print("-" * 100)
            print("constitutive_params.params \n", constitutive_params.params)
            print("constitutive_params.raw_params.grad \n", constitutive_params.raw_params.grad)
            print("-" * 100)

        if cfg.show_plots and epoch % cfg.field_plot_every == 0:
            show_field_diagnostics(compute_diagnostics(out, ctx), mesh, data.displacements, epoch)
            plt.close()

    return start_time


def train_stage2(model, constitutive_params, ctx, loss_history, theta_true, mesh, data, cfg, model_symbol,
                 start_time):
    """Stage 2: alternate L-BFGS on the network and Lp sparse regression on theta."""
    for iteration in range(1, cfg.ado_iterations + 1):
        # (a) L-BFGS on the network, theta fixed
        optimizer = torch.optim.LBFGS(
            list(model.parameters()),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            history_size=cfg.lbfgs_history_size,
        )
        eval_count = 0

        def closure():
            nonlocal eval_count
            eval_count += 1
            optimizer.zero_grad()

            model.train()
            out = compute_losses(model, constitutive_params, ctx, loss_history)
            loss = out.pinn_loss
            loss.backward()

            constitutive_params.save_param_history()

            if eval_count % cfg.print_every == 0:
                elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"L-BFGS solver : Epoch {eval_count} \t PINN Loss = {loss:.4e} \t Elapsed Time = {elapsed_time_str}")
            if cfg.show_plots and eval_count % cfg.plot_every == 0:
                plot_loss_history(loss_history)
                plot_params(constitutive_params.params, theta_true, eval_count)
            if cfg.show_plots and eval_count % cfg.field_plot_every == 0:
                show_field_diagnostics(compute_diagnostics(out, ctx), mesh, data.displacements, eval_count)
            return loss

        optimizer.step(closure)

        # (b) Sparse regression on theta, network fixed
        best_theta = main_Lp(model, constitutive_params, ctx, loss_history, cfg.lambda_factor(model_symbol),
                             num_guesses=cfg.num_initial_guesses, p=cfg.lp_p, threshold=cfg.lp_threshold)
        if cfg.show_plots:
            plot_params(best_theta, theta_true, iteration)
        constitutive_params.update_constitutive_params(best_theta)
        constitutive_params.save_param_history()


def save_results(model, constitutive_params, loss_history, theta_true, output_dir, tag):
    """Write the figures, loss/coefficient histories and the checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    save_params_plot(constitutive_params.params, theta_true, -1, os.path.join(output_dir, f'Params_{tag}.jpg'))
    save_loss_history_plot(loss_history, os.path.join(output_dir, f'Loss_{tag}.jpg'))

    np.save(os.path.join(output_dir, f'Loss_{tag}.npy'), np.array(list(loss_history.items()), dtype=object))
    # The misspelled file name is kept for compatibility with existing results.
    np.save(os.path.join(output_dir, f'Consitutive_parameter_{tag}.npy'), np.array(constitutive_params.param_history))

    torch.save({
        'model_state_dict': model.state_dict(),
        'constitutive_params_state_dict': constitutive_params.state_dict(),
    }, os.path.join(output_dir, f'AI_Model_{tag}'))


def run(model_symbol, noise_percent, cfg, data_root, output_dir, device):
    """One discovery run for one benchmark model and noise level."""
    theta_true = true_theta(model_symbol)
    set_seed(cfg.seed)

    # Mesh and measurement points
    data_path = os.path.join(data_root, model_symbol)
    mesh = load_mesh(os.path.join(data_path, 'Input_DG.inp'))
    data_known_idx = select_measurement_nodes(mesh.node_coords, cfg.hole_radius, cfg.measurement_angles_deg)
    if cfg.show_plots:
        plot_mesh(mesh.node_coords, mesh.elements, data_known_idx)

    # FEM data with measurement noise
    data = load_fem_data(data_path)
    true_totals = total_reaction_forces(data)
    add_noise(data, data_known_idx, noise_percent)
    if cfg.show_plots:
        plot_reaction_force_noise(true_totals, total_reaction_forces(data))

    ctx = build_loss_context(mesh, data, data_known_idx, cfg.hole_radius, cfg.near_hole_margin,
                             cfg.data_loss_weight, device, cfg.num_increments)

    # Models (creation order fixes the random initialization)
    model = PINN(width=cfg.width, f33=cfg.f33).to(device)
    constitutive_params = ConstitutiveParams(device, NUM_TERMS)
    loss_history = new_loss_history()

    # Initial evaluation; it is recorded as the first loss-history entry.
    compute_losses(model, constitutive_params, ctx, loss_history)

    start_time = train_stage1(model, constitutive_params, ctx, loss_history, theta_true, mesh, data, cfg, device)
    train_stage2(model, constitutive_params, ctx, loss_history, theta_true, mesh, data, cfg, model_symbol, start_time)

    tag = f'{model_symbol}_Noise_{noise_percent}_Trial_{cfg.trial}'
    save_results(model, constitutive_params, loss_history, theta_true, output_dir, tag)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='all', choices=MODEL_SYMBOLS + ['all'],
                        help='benchmark model to discover')
    parser.add_argument('--noise', default='all',
                        help=f"noise level (e.g. 0.01) or 'all' = {NOISE_LEVELS}")
    parser.add_argument('--epochs-adam', type=int, default=Config.num_epochs_adam, help='Stage 1 Adam epochs')
    parser.add_argument('--ado-iters', type=int, default=Config.ado_iterations,
                        help='Stage 2 cycles of L-BFGS + sparse regression')
    parser.add_argument('--lbfgs-max-iter', type=int, default=Config.lbfgs_max_iter,
                        help='L-BFGS iterations per Stage 2 cycle')
    parser.add_argument('--num-guesses', type=int, default=Config.num_initial_guesses,
                        help='sparse regression: rounds of 3 random initial guesses')
    parser.add_argument('--seed', type=int, default=Config.seed, help='torch random seed')
    parser.add_argument('--output-dir', default=os.path.join(BASE_DIR, 'Model_Save'), help='output folder')
    parser.add_argument('--show-plots', action='store_true', help='show intermediate diagnostic figures')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(seed=args.seed, num_epochs_adam=args.epochs_adam, ado_iterations=args.ado_iters,
                 lbfgs_max_iter=args.lbfgs_max_iter, num_initial_guesses=args.num_guesses,
                 show_plots=args.show_plots)
    if not cfg.show_plots:
        plt.switch_backend('Agg')  # figures are only saved

    model_symbols = MODEL_SYMBOLS if args.model == 'all' else [args.model]
    noise_list = NOISE_LEVELS if args.noise == 'all' else [float(args.noise)]
    data_root = os.path.join(REPO_DIR, 'FEM_Data', *cfg.data_subdir.split('/'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_symbol in model_symbols:
        for noise_percent in noise_list:
            print(f"===== Model {model_symbol}, noise {noise_percent} =====")
            run(model_symbol, noise_percent, cfg, data_root, args.output_dir, device)


if __name__ == '__main__':
    main()
