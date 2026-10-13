"""Weak-form PINN loss.

Assembles the linear system A(u) theta = F of the weak-form force balance:
A collects, for every degree of freedom (x DOFs first, then y DOFs), the
internal nodal forces of each candidate term, F holds the measured reaction
forces. The PINN loss combines

  * Dirichlet/data loss: boundary displacements of the four edges and the
    in-plane displacements of the measurement points (weight 1/2),
  * weak-form residual of the free DOFs (zero net force),
  * weak-form residual of the free DOFs near the hole.

The residual of the resultant edge forces is logged (and enters the
coefficient loss) but not the PINN loss.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .library import compute_analytic_dQ_dF, compute_invariants


@dataclass
class LossContext:
    """Fixed tensors and index sets used by every loss evaluation."""
    num_nodes: int
    num_increments: int
    # Node indices (0-based) and DOF indices (x DOF = node, y DOF = node + num_nodes)
    top_idx: np.ndarray
    bot_idx: np.ndarray
    right_idx: np.ndarray
    left_idx: np.ndarray
    hole_idx: np.ndarray
    data_idx: np.ndarray
    top_idx_y: np.ndarray
    bot_idx_y: np.ndarray
    right_idx_x: np.ndarray
    left_idx_x: np.ndarray
    fix_idx: np.ndarray          # DOFs with prescribed displacement
    free_idx: np.ndarray         # free DOFs away from the hole (PINN loss)
    near_hole_idx: np.ndarray    # free DOFs within hole_radius + margin of the origin
    free_idx_all: np.ndarray     # all free DOFs (sparse regression)
    # Mesh tensors
    connectivity: torch.Tensor   # (n_elements, 3) 0-based node indices
    B_matrices: torch.Tensor     # (n_elements, 4, 6)
    element_areas: torch.Tensor  # (n_elements,)
    global_dof_index: torch.Tensor  # (n_elements, 6) global DOF of each element DOF
    # Network inputs for all nodes and increments
    X_nodes: torch.Tensor        # (num_increments * n_nodes, 2)
    time_nodes: torch.Tensor     # (num_increments * n_nodes, 1)
    # Measured displacements, increments 1..10
    U_true_top_y: torch.Tensor
    U_true_bot_y: torch.Tensor
    U_true_right_x: torch.Tensor
    U_true_left_x: torch.Tensor
    U_true_data_x: torch.Tensor
    U_true_data_y: torch.Tensor
    U_ref: float                 # displacement scale (max top-edge displacement)
    F_external: torch.Tensor     # (num_increments, 2 * n_nodes, 1) unnormalized
    data_loss_weight: float


def new_loss_history():
    return {
        "dirichlet_loss": [],
        "weak_form_loss_free": [],
        "weak_form_loss_fix": [],
        "weak_form_loss_near_hole": [],
    }


def compute_F_vector(num_nodes, top_line_rf, bot_line_rf, right_line_rf, left_line_rf, device, num_increments=10):
    """External nodal force vector (num_increments, 2 * num_nodes, 1) from the edge reaction forces.

    Frame 0 (initial state) is dropped; nodes not on an edge get zero force.
    """
    top_line_rf = torch.tensor(top_line_rf, dtype=torch.float32, device=device)
    bot_line_rf = torch.tensor(bot_line_rf, dtype=torch.float32, device=device)
    right_line_rf = torch.tensor(right_line_rf, dtype=torch.float32, device=device)
    left_line_rf = torch.tensor(left_line_rf, dtype=torch.float32, device=device)

    combined_rf = torch.cat([top_line_rf, bot_line_rf, right_line_rf, left_line_rf], dim=1)

    # Sort by node label
    sorted_values, sorted_indices = torch.sort(combined_rf[:, :, 0], dim=1)
    combined_rf = combined_rf.gather(1, sorted_indices.unsqueeze(-1).expand_as(combined_rf))

    node_indices = combined_rf[0, :, 0].int() - 1
    reaction_forces = torch.zeros((num_increments, num_nodes, 3), dtype=torch.float32, device=device)
    reaction_forces[:, :, 0] = torch.arange(1, num_nodes + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_increments, 1)
    reaction_forces[:, node_indices, 1:] = combined_rf[1:, :, 1:].clone()

    external_force_x = reaction_forces[:, :, 1].clone()
    external_force_y = reaction_forces[:, :, 2].clone()
    return torch.cat([external_force_x, external_force_y], dim=1).unsqueeze(-1)


def build_loss_context(mesh, data, data_known_idx, hole_radius, near_hole_margin, data_loss_weight, device,
                       num_increments=10):
    """Precompute index sets and tensors for the loss (depends only on mesh and data)."""
    nodes = mesh.nodes
    num_nodes = len(nodes)
    displacements = data.displacements

    top_idx_y = mesh.top_idx + num_nodes
    bot_idx_y = mesh.bot_idx + num_nodes
    right_idx_x = mesh.right_idx
    left_idx_x = mesh.left_idx

    total_idx = np.arange(2 * num_nodes)
    fix_idx = np.unique(np.concatenate((top_idx_y, bot_idx_y, right_idx_x, left_idx_x)))
    free_idx_all = np.setdiff1d(total_idx, fix_idx)

    # Free DOFs whose node lies within hole_radius + margin of the hole center
    free_idx_temp = np.where(free_idx_all < num_nodes, free_idx_all, free_idx_all - num_nodes)
    target_point = np.array([0, 0])
    free_node_coords = nodes[free_idx_temp]
    distances = np.sqrt((free_node_coords[:, 1] - target_point[0]) ** 2 + (free_node_coords[:, 2] - target_point[1]) ** 2)
    near_hole_idx = free_idx_all[(distances <= hole_radius + near_hole_margin)]
    free_idx = np.setdiff1d(free_idx_all, near_hole_idx)

    # Mesh tensors
    elements_t = torch.tensor(mesh.elements, dtype=torch.long, device=device)
    connectivity = elements_t[:, 1:] - 1
    num_elements = len(elements_t)
    global_dof_index = torch.zeros((num_elements, 6), dtype=torch.long, device=device)
    global_dof_index[:, 0:3] = connectivity                  # x DOFs
    global_dof_index[:, 3:6] = connectivity + num_nodes * 1  # y DOFs

    def as_tensor(array):
        return torch.tensor(array, dtype=torch.float32, device=device)

    X_nodes = torch.tensor(nodes[:, 1:], dtype=torch.float32).to(device).repeat(num_increments, 1, 1)
    time_nodes = torch.arange(1, num_increments + 1, dtype=torch.float32, device=device).view(-1, 1).repeat(1, X_nodes.size(1))

    return LossContext(
        num_nodes=num_nodes,
        num_increments=num_increments,
        top_idx=mesh.top_idx, bot_idx=mesh.bot_idx, right_idx=mesh.right_idx, left_idx=mesh.left_idx,
        hole_idx=mesh.hole_idx, data_idx=data_known_idx,
        top_idx_y=top_idx_y, bot_idx_y=bot_idx_y, right_idx_x=right_idx_x, left_idx_x=left_idx_x,
        fix_idx=fix_idx, free_idx=free_idx, near_hole_idx=near_hole_idx, free_idx_all=free_idx_all,
        connectivity=connectivity,
        B_matrices=as_tensor(mesh.B_matrices),
        element_areas=as_tensor(mesh.element_areas)[:, 1],
        global_dof_index=global_dof_index,
        X_nodes=X_nodes.reshape(-1, 2),
        time_nodes=time_nodes.reshape(-1, 1),
        U_true_top_y=as_tensor(displacements[1:, mesh.top_idx, 2]),
        U_true_bot_y=as_tensor(displacements[1:, mesh.bot_idx, 2]),
        U_true_right_x=as_tensor(displacements[1:, mesh.right_idx, 1]),
        U_true_left_x=as_tensor(displacements[1:, mesh.left_idx, 1]),
        U_true_data_x=as_tensor(displacements[1:, data_known_idx, 1]),
        U_true_data_y=as_tensor(displacements[1:, data_known_idx, 2]),
        U_ref=displacements[1:, mesh.top_idx, 1:].max(),
        F_external=compute_F_vector(num_nodes, data.top_rf, data.bot_rf, data.right_rf, data.left_rf,
                                    device, num_increments),
        data_loss_weight=data_loss_weight,
    )


def compute_deformation_gradient(ctx, U_pred):
    """Element deformation gradients (num_increments, n_elements, 3, 3) from nodal displacements.

    U_pred: (num_increments, n_nodes, 3). F33 = 1 + mean of the element's u_z.
    """
    num_increments = ctx.num_increments
    U_pred = U_pred.view(num_increments, ctx.num_nodes, 3)
    element_displacements = U_pred[:, ctx.connectivity, :]  # (n_increments, n_elements, 3, 3)

    x_displacements = element_displacements[..., 0]
    y_displacements = element_displacements[..., 1]
    z_displacements = element_displacements[..., 2]
    element_displacements_flat = torch.cat([x_displacements, y_displacements], dim=-1).unsqueeze(-1)  # (.., 6, 1)

    B_matrices_expanded = ctx.B_matrices.unsqueeze(0).expand(num_increments, -1, -1, -1)  # (.., 4, 6)
    F_flat = torch.einsum('eijk,eikl->eijl', B_matrices_expanded, element_displacements_flat).squeeze(-1)  # (.., 4)

    F = torch.zeros(F_flat.shape[0], F_flat.shape[1], 3, 3, device=U_pred.device)
    F[:, :, 0, 0] = F_flat[:, :, 0].clone()  # F11
    F[:, :, 1, 1] = F_flat[:, :, 1].clone()  # F22
    F[:, :, 0, 1] = F_flat[:, :, 2].clone()  # F12
    F[:, :, 1, 0] = F_flat[:, :, 3].clone()  # F21
    F[:, :, 2, 2] = torch.sum(z_displacements, dim=-1) / 3  # F33

    identity_matrix = torch.eye(3, device=U_pred.device).unsqueeze(0).unsqueeze(0)
    identity_matrix = identity_matrix.expand(F.size(0), F.size(1), -1, -1)
    return F + identity_matrix


def compute_A_matrix(ctx, dQ_dF):
    """Global matrix A (num_increments, 2 * n_nodes, n_terms) of internal nodal forces per term."""
    num_increments = ctx.num_increments
    B_matrices_expanded = ctx.B_matrices.unsqueeze(0).expand(num_increments, -1, -1, -1)  # (.., 4, 6)

    # In-plane components of dQ/dF: (n_increments, n_elements, n_terms, 4)
    dQ_dF_vectorized = torch.stack([dQ_dF[..., 0, 0],  # F11
                                    dQ_dF[..., 1, 1],  # F22
                                    dQ_dF[..., 0, 1],  # F12
                                    dQ_dF[..., 1, 0],  # F21
                                    ], dim=-1)

    elem_A = torch.einsum('eikj,eilk->eijl', B_matrices_expanded, dQ_dF_vectorized)  # (.., 6, n_terms)
    elem_A = elem_A * ctx.element_areas.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    global_A = torch.zeros((num_increments, ctx.num_nodes * 2, dQ_dF.size(2)), dtype=elem_A.dtype, device=dQ_dF.device)
    index_tensor = ctx.global_dof_index.unsqueeze(2).expand(-1, -1, elem_A.size(3))  # (n_elements, 6, n_terms)
    for inc in range(num_increments):
        global_A[inc].scatter_add_(0, index_tensor.view(-1, index_tensor.size(2)), elem_A[inc].view(-1, elem_A.size(3)))
    return global_A


def normalize_system(A, F, ctx):
    """Scale A and F in place: x rows by F_y/F_x per increment, then each increment by max(F_y)/F_y."""
    n = ctx.num_nodes
    F_y = torch.sum(F[:, ctx.top_idx_y], dim=1).view(-1)
    F_x = torch.sum(F[:, ctx.right_idx_x], dim=1).view(-1)

    for i in range(ctx.num_increments):
        A[i, :n] = A[i, :n] * F_y[i] / F_x[i]
        F[i, :n] = F[i, :n] * F_y[i] / F_x[i]

    for i in range(ctx.num_increments):
        A[i, :] = A[i, :] * F_y.max() / F_y[i].max()
        F[i, :] = F[i, :] * F_y.max() / F_y[i].max()
    return F_y


@dataclass
class LossOutput:
    pinn_loss: torch.Tensor          # loss for the network
    constitutive_loss: torch.Tensor  # unscaled weak-form residual, loss for theta
    U_pred: torch.Tensor             # (num_increments, n_nodes, 3), detached
    deform_gradient: torch.Tensor    # (num_increments, n_elements, 3, 3), detached
    dQ_dF: torch.Tensor              # detached
    A: torch.Tensor                  # normalized A, detached (shares memory with the graph tensor)
    F: torch.Tensor                  # normalized F, detached
    theta: torch.Tensor              # (n_terms, 1) snapshot of the coefficients used


def compute_losses(model, constitutive_params, ctx, loss_history):
    """Evaluate the PINN and constitutive losses and append the loss terms to loss_history."""
    num_increments = ctx.num_increments

    # Prediction
    U_pred_nodes = model(ctx.X_nodes, ctx.time_nodes).reshape(num_increments, -1, 3)
    U_pred_top = U_pred_nodes[:, ctx.top_idx, :]
    U_pred_bot = U_pred_nodes[:, ctx.bot_idx, :]
    U_pred_right = U_pred_nodes[:, ctx.right_idx, :]
    U_pred_left = U_pred_nodes[:, ctx.left_idx, :]
    U_pred_data = U_pred_nodes[:, ctx.data_idx, :]

    U_pred_top_y = U_pred_top[:, :, 1].clone()
    U_pred_bot_y = U_pred_bot[:, :, 1].clone()
    U_pred_right_x = U_pred_right[:, :, 0].clone()
    U_pred_left_x = U_pred_left[:, :, 0].clone()
    U_pred_data_x = U_pred_data[:, :, 0].clone()
    U_pred_data_y = U_pred_data[:, :, 1].clone()

    # Dirichlet (boundary) and data loss
    dirichlet_loss = torch.sum((ctx.U_true_top_y - U_pred_top_y) ** 2)
    dirichlet_loss += torch.sum((ctx.U_true_bot_y - U_pred_bot_y) ** 2)
    dirichlet_loss += torch.sum((ctx.U_true_right_x - U_pred_right_x) ** 2)
    dirichlet_loss += torch.sum((ctx.U_true_left_x - U_pred_left_x) ** 2)
    dirichlet_loss += torch.sum((ctx.U_true_data_x - U_pred_data_x) ** 2) * ctx.data_loss_weight
    dirichlet_loss += torch.sum((ctx.U_true_data_y - U_pred_data_y) ** 2) * ctx.data_loss_weight

    library_params = constitutive_params.params.view(-1, 1)

    # Weak form
    deform_gradient = compute_deformation_gradient(ctx, U_pred_nodes)
    dQ_dF = compute_analytic_dQ_dF(deform_gradient)
    A = compute_A_matrix(ctx, dQ_dF)
    F = ctx.F_external.clone()
    F_y = normalize_system(A, F, ctx)

    error_total = torch.matmul(A, library_params) - F

    weak_form_loss_free = torch.sum(error_total[:, ctx.free_idx, :] ** 2)

    weak_form_loss_fix = torch.sum(torch.sum(error_total[:, ctx.top_idx_y], dim=1) ** 2)
    weak_form_loss_fix += torch.sum(torch.sum(error_total[:, ctx.bot_idx_y], dim=1) ** 2)
    weak_form_loss_fix += torch.sum(torch.sum(error_total[:, ctx.right_idx_x], dim=1) ** 2)
    weak_form_loss_fix += torch.sum(torch.sum(error_total[:, ctx.left_idx_x], dim=1) ** 2)

    weak_form_loss_near_hole = torch.sum(error_total[:, ctx.near_hole_idx, :] ** 2)

    # Coefficient loss: unscaled weak-form residual
    constitutive_loss = weak_form_loss_free + weak_form_loss_fix + weak_form_loss_near_hole

    # Scaled PINN loss
    F_ref = F_y.max()
    dirichlet_loss = dirichlet_loss / ctx.U_ref ** 2
    weak_form_loss_free = weak_form_loss_free / F_ref ** 2
    weak_form_loss_fix = weak_form_loss_fix / F_ref ** 2
    weak_form_loss_near_hole = weak_form_loss_near_hole / F_ref ** 2

    pinn_loss = dirichlet_loss + weak_form_loss_free + weak_form_loss_near_hole

    if torch.isnan(dirichlet_loss + weak_form_loss_free + weak_form_loss_fix + weak_form_loss_near_hole).any():
        print("-" * 60)
        print("dirichlet_loss : ", dirichlet_loss)
        print("weak_form_loss_free : ", weak_form_loss_free)
        print("weak_form_loss_fix : ", weak_form_loss_fix)
        print("weak_form_loss_near_hole : ", weak_form_loss_near_hole)

    loss_history["dirichlet_loss"].append(dirichlet_loss.item())
    loss_history["weak_form_loss_free"].append(weak_form_loss_free.item())
    loss_history["weak_form_loss_fix"].append(weak_form_loss_fix.item())
    loss_history["weak_form_loss_near_hole"].append(weak_form_loss_near_hole.item())

    return LossOutput(
        pinn_loss=pinn_loss,
        constitutive_loss=constitutive_loss,
        U_pred=U_pred_nodes.detach(),
        deform_gradient=deform_gradient.detach(),
        dQ_dF=dQ_dF.detach(),
        A=A.detach(),
        F=F.detach(),
        theta=library_params.detach().clone(),
    )


def compute_diagnostics(out, ctx):
    """Fields used by the diagnostic plots (stresses, invariants, resultant forces, residual)."""
    library_params = out.theta
    deform_gradient = out.deform_gradient
    internal_force = torch.matmul(out.A, library_params)
    external_force = out.F
    J, I1_bar, I2_bar = compute_invariants(deform_gradient)
    PK1_stress = torch.sum(out.dQ_dF * library_params.view(1, 1, library_params.size(0), 1, 1), dim=2)
    cauchy_stress = torch.matmul(PK1_stress, deform_gradient.transpose(-2, -1)) / J.view(J.size(0), J.size(1), 1, 1)

    def edge_sum(force, idx):
        return torch.sum(force[:, idx, :], dim=1).view(-1)

    return {
        'U_pred': out.U_pred,
        'DG': deform_gradient,
        'J': J, 'I1_bar': I1_bar, 'I2_bar': I2_bar,
        'cauchy_stress': cauchy_stress,
        'error': torch.abs(internal_force - external_force),
        'forces': (edge_sum(internal_force, ctx.top_idx_y), edge_sum(external_force, ctx.top_idx_y),
                   edge_sum(internal_force, ctx.bot_idx_y), edge_sum(external_force, ctx.bot_idx_y),
                   edge_sum(internal_force, ctx.right_idx_x), edge_sum(external_force, ctx.right_idx_x),
                   edge_sum(internal_force, ctx.left_idx_x), edge_sum(external_force, ctx.left_idx_x)),
    }
