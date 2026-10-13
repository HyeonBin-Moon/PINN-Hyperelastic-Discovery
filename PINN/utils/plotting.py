"""Plotting.

`plot_*` functions show diagnostic figures during training (only with
--show-plots); `save_params_plot` and `save_loss_history_plot` write the final
figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from .data import fit_with_constraint
from .library import invariant_derivatives


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def plot_mesh(node_coords, elements, data_known_idx):
    """Mesh with the displacement measurement points highlighted."""
    plt.figure(figsize=(6, 6))
    for elem in elements[:, 1:]:
        polygon = node_coords[elem - 1]
        plt.fill(polygon[:, 0], polygon[:, 1], edgecolor='k', fill=False)

    plt.scatter(node_coords[data_known_idx, 0], node_coords[data_known_idx, 1], color='red', s=50, edgecolor='k', label='Data Known Points')
    plt.legend()
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Mesh with Data Known Points Highlighted')
    plt.show()


def plot_reaction_force_noise(true_totals, noisy_totals, degree=4):
    """True, noisy and polynomial-filtered resultant forces of the four edges."""
    top_filtered = fit_with_constraint(noisy_totals[0], degree)
    right_filtered = fit_with_constraint(noisy_totals[2], degree)
    filtered = (top_filtered, -top_filtered.copy(), right_filtered, -right_filtered.copy())
    titles = ("Top Line RF Total Force", "Bottom Line RF Total Force",
              "Right Line RF Total Force", "Left Line RF Total Force")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, true, noisy, filt, title in zip(axes, true_totals, noisy_totals, filtered, titles):
        ax.plot(true, label="True Data", color="green")
        ax.plot(noisy, label="Noisy Data", color="blue")
        ax.plot(filt, label="Filtered Data (Constrained Polyfit)", color="orange")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Training diagnostics
# ----------------------------------------------------------------------------

def _draw_params(params, true_params, epoch):
    """Bar plot of predicted vs. true coefficients with the 5 largest labelled."""
    top_n = 5
    params_np = params.detach().cpu().numpy()
    true_params_np = true_params.detach().cpu().numpy()
    x_indices = range(1, len(params_np) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.bar(x_indices, params_np, label='Predicted Params', alpha=0.8, color='tab:blue')
    top_indices = torch.topk(torch.tensor(params_np), top_n).indices.numpy()
    for idx in top_indices:
        ax.text(idx + 1, params_np[idx], f'{idx + 1}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.bar(x_indices, true_params_np, label='True Params', alpha=0.5, color='tab:red', linewidth=2)

    ax.set_title(f'Constitutive Parameters at Epoch {epoch}')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Value')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()


def _draw_loss_history(loss_history, ylim):
    """Log-scale loss history (all series truncated to the shortest one)."""
    min_length = min(len(values) for values in loss_history.values())
    plt.figure(figsize=(8, 6))
    for key, values in loss_history.items():
        values = np.array(values[:min_length])
        plt.plot(range(len(values)), values, label=f"{key.replace('_', ' ').title()}", alpha=0.5)
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Loss History")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(*ylim)


def plot_params(params, true_params, epoch):
    _draw_params(params, true_params, epoch)
    plt.show()
    plt.close()


def plot_loss_history(loss_history):
    _draw_loss_history(loss_history, ylim=(1e-10, 1e3))
    plt.show()
    plt.close()


def _scatter_field(fig, ax, x, y, values, title, label=None, s=20, contour_grid=None):
    sc = ax.scatter(x, y, c=values, cmap='jet', marker='o', s=s)
    if contour_grid is not None:
        xi, yi = contour_grid
        zi = griddata((x, y), values, (xi, yi), method='linear')
        ax.contour(xi, yi, zi, colors='black', linewidths=0.5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if label is not None:
        ax.grid(True)
        fig.colorbar(sc, ax=ax, label=label)
    else:
        fig.colorbar(sc, ax=ax)


def plot_displacement(U_pred, U_true, node_coords, epoch=None, deformed=False, plot_increment=10, dpi=100):
    """Predicted displacement components (with iso-lines) and absolute errors."""
    k = plot_increment - 1
    if deformed:
        node_coords = node_coords.copy() + U_pred[k, :, :2]
    x, y = node_coords[:, 0], node_coords[:, 1]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi)
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    grid = np.meshgrid(xi, yi)

    for c, name in enumerate(('u_x', 'u_y', 'u_z')):
        _scatter_field(fig, axs[0, c], x, y, U_pred[k, :, c], f'{name} Displacement at Epoch {epoch}',
                       label=f'{name} Displacement', contour_grid=grid)
    for c, name in enumerate(('u_x', 'u_y')):
        error = np.abs(U_pred[k, :, c] - U_true[k, :, c])
        _scatter_field(fig, axs[1, c], x, y, error, f'{name} Error at Epoch {epoch}', label=f'{name} Error')

    plt.suptitle(f'Displacement Prediction and Error Fields at Epoch {epoch}')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_invariants(I1_bar, I2_bar, J, gaussian_points, epoch=None, plot_increment=10, dpi=100):
    """I1_bar, I2_bar and J at the element centroids."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)
    x, y = gaussian_points[:, 1], gaussian_points[:, 2]
    for ax, values, name in zip(axs, (I1_bar, I2_bar, J), ('I1_bar', 'I2_bar', 'J')):
        _scatter_field(fig, ax, x, y, values[plot_increment - 1], f'{name} at Epoch {epoch}', label=name)
    plt.suptitle(f'Invariants Field at Epoch {epoch}')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_external_force(Top_F_pred, Top_F_true, Bot_F_pred, Bot_F_true, Right_F_pred, Right_F_true,
                        Left_F_pred, Left_F_true, dpi=100):
    """Predicted (internal) vs. measured resultant edge forces per increment."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), dpi=dpi)
    increments = np.arange(1, 11)
    pairs = ((Top_F_pred, Top_F_true, 1, "Top Y-dir"), (Bot_F_pred, Bot_F_true, -1, "Bot Y-dir"),
             (Right_F_pred, Right_F_true, 1, "Right X-dir"), (Left_F_pred, Left_F_true, -1, "Left X-dir"))
    for ax, (pred, true, sign, title) in zip(axs, pairs):
        ax.plot(increments, sign * pred.cpu().detach().numpy(), '-o', alpha=0.5)
        ax.plot(increments, sign * true.cpu().detach().numpy(), '-o', alpha=0.5)
        ax.set_title(title)
        ax.grid()
    plt.show()
    plt.close()


def plot_residual(node_coords, error, epoch=None, plot_increment=10, dpi=100):
    """Nodal weak-form residual |A theta - F| of the x and y DOFs."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)
    n = len(node_coords)
    error_x = error[plot_increment - 1, :n].reshape(-1)
    error_y = error[plot_increment - 1, n:].reshape(-1)
    x, y = node_coords[:, 0], node_coords[:, 1]
    _scatter_field(fig, axs[0], x, y, error_x, f'X Residual at Epoch {epoch}', label='X Residual', s=30)
    _scatter_field(fig, axs[1], x, y, error_y, f'Y Residual at Epoch {epoch}', label='Y Residual', s=30)
    plt.show()
    plt.close()


def plot_tensor_components(gaussian_points, tensor, name, epoch=None, plot_increment=10, dpi=100):
    """Components 11, 12, 21, 22, 33 of a (n_increments, n_elements, 3, 3) field at the centroids."""
    fig, axs = plt.subplots(1, 5, figsize=(25, 5), dpi=dpi)
    s_indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    x, y = gaussian_points[:, 1], gaussian_points[:, 2]
    values = tensor[plot_increment - 1]
    for ax, (a, b) in zip(axs, s_indices):
        _scatter_field(fig, ax, x, y, values[:, a, b], f'{name}{a + 1}{b + 1} at Epoch {epoch}', s=5)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_d_invariants_dF(I1_bar, I2_bar, J, DG, gaussian_points, epoch=None, plot_increment=10, dpi=100):
    """Components of dI1_bar/dF, dI2_bar/dF and dJ/dF."""
    dI1_bar_dF, dI2_bar_dF, dJ_dF = invariant_derivatives(DG, J, I1_bar, I2_bar)
    for values, name in ((dI1_bar_dF, 'dI1_dF '), (dI2_bar_dF, 'dI2_dF '), (dJ_dF, 'dJ_dF ')):
        plot_tensor_components(gaussian_points, values.cpu().detach().numpy(), name, epoch, plot_increment, dpi)


def show_field_diagnostics(diag, mesh, displacements, epoch):
    """All field plots of a training checkpoint (fields from loss.compute_diagnostics)."""
    def to_np(t):
        return t.cpu().detach().numpy()

    num_nodes = mesh.num_nodes
    U_true = np.concatenate([displacements[1:, :, 1:], np.zeros((10, num_nodes, 1))], axis=2)
    plot_invariants(to_np(diag['I1_bar']), to_np(diag['I2_bar']), to_np(diag['J']), mesh.gaussian_points, epoch=epoch)
    plot_displacement(to_np(diag['U_pred']), U_true, mesh.node_coords, epoch=epoch, deformed=True)
    plot_external_force(*diag['forces'])
    plot_residual(mesh.node_coords, to_np(diag['error']), epoch=epoch)
    plot_tensor_components(mesh.gaussian_points, to_np(diag['DG']), 'DG', epoch=epoch)
    plot_tensor_components(mesh.gaussian_points, to_np(diag['cauchy_stress']), 'S', epoch=epoch)
    plot_d_invariants_dF(diag['I1_bar'], diag['I2_bar'], diag['J'], diag['DG'], mesh.gaussian_points, epoch=epoch)


# ----------------------------------------------------------------------------
# Saved figures
# ----------------------------------------------------------------------------

def save_params_plot(params, true_params, epoch, save_path):
    """Save the coefficient bar plot to save_path."""
    _draw_params(params, true_params, epoch)
    plt.savefig(save_path)
    plt.close()


def save_loss_history_plot(loss_history, save_path):
    """Save the loss history plot to save_path."""
    _draw_loss_history(loss_history, ylim=(1e-8, 1e4))
    plt.savefig(save_path)
    plt.close()
