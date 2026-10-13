"""Lp-regularized (p < 1) thresholded sparse regression for the library coefficients.

With the network fixed, theta minimizes ||AA theta - FF||^2 + lam * sum |theta_k|^p
by iteratively reweighted least squares with hard thresholding, started from
many random initial guesses; the converged solution with the smallest residual
is kept.
"""
import torch
from tqdm import tqdm

from .loss import compute_losses


def generate_A_and_F(model, constitutive_params, ctx, loss_history):
    """Regression system from the current network: rows of the free DOFs plus the four edge resultants.

    Evaluates the loss once (this appends an entry to loss_history), takes the
    normalized A and F of that evaluation and normalizes them once more, as in
    the original script.
    """
    out = compute_losses(model, constitutive_params, ctx, loss_history)
    A_matrix, F_vector = out.A, out.F
    n = ctx.num_nodes

    F_y = torch.sum(F_vector[:, ctx.top_idx_y], dim=1).view(-1)
    F_x = torch.sum(F_vector[:, ctx.right_idx_x], dim=1).view(-1)

    # Scale x rows
    for i in range(ctx.num_increments):
        A_matrix[i, :n] = A_matrix[i, :n] * F_y[i] / F_x[i]
        F_vector[i, :n] = F_vector[i, :n] * F_y[i] / F_x[i]

    # Scale increments
    F_y_max = F_y.max()
    for i in range(ctx.num_increments):
        A_matrix[i, :] = A_matrix[i, :] * F_y_max / F_y[i].max()
        F_vector[i, :] = F_vector[i, :] * F_y_max / F_y[i].max()

    free_idx = ctx.free_idx_all
    A_free, F_free = A_matrix[:, free_idx], F_vector[:, free_idx]
    A_fix1, F_fix1 = torch.sum(A_matrix[:, ctx.top_idx_y], dim=1), torch.sum(F_vector[:, ctx.top_idx_y], dim=1)
    A_fix2, F_fix2 = torch.sum(A_matrix[:, ctx.bot_idx_y], dim=1), torch.sum(F_vector[:, ctx.bot_idx_y], dim=1)
    A_fix3, F_fix3 = torch.sum(A_matrix[:, ctx.right_idx_x], dim=1), torch.sum(F_vector[:, ctx.right_idx_x], dim=1)
    A_fix4, F_fix4 = torch.sum(A_matrix[:, ctx.left_idx_x], dim=1), torch.sum(F_vector[:, ctx.left_idx_x], dim=1)
    A_fix = torch.stack([A_fix1, A_fix2, A_fix3, A_fix4], dim=1)
    F_fix = torch.stack([F_fix1, F_fix2, F_fix3, F_fix4], dim=1)

    AA, FF = torch.concat([A_free, A_fix], dim=1).detach().cpu(), torch.concat([F_free, F_fix], dim=1).detach().cpu()
    AA = AA.reshape(-1, AA.size(-1))
    FF = FF.reshape(-1, 1)
    return AA, FF


def compute_residual(A, b, theta):
    """Sum of squared residuals ||A theta - b||^2."""
    theta = theta.reshape(-1, 1)
    b = b.reshape(-1, 1)
    residual = torch.abs(A @ theta - b)
    return torch.sum(residual ** 2)


def apply_Lp_threshold_regression(AT_A, AT_b, theta, p=1, lam=1e-5, threshold=1e-2, max_iter=100, tol=1e-5):
    """Iteratively reweighted least squares for the Lp penalty with hard thresholding.

    AT_A: (n, n), AT_b: (n, 1), theta: (n,) initial guess.
    Each iteration zeroes theta_k < threshold and solves
    (AT_A + p lam diag(|theta|^(p-2))) theta = AT_b on the active set.
    Returns (theta, converged).
    """
    theta = theta.clone().reshape(-1)
    AT_b = AT_b.reshape(-1)

    LHS = AT_A.clone()
    RHS = AT_b.clone()

    converged = False
    for _ in range(max_iter):
        temp = theta.clone().detach()
        theta_greater = theta.clone() >= threshold
        theta[~theta_greater] = 0
        LHS_Lp = LHS[theta_greater, :][:, theta_greater] + p * lam * torch.diag(torch.abs(theta[theta_greater]) ** (p - 2))
        try:
            theta[theta_greater] = torch.linalg.solve(LHS_Lp, RHS[theta_greater])
        except Exception:
            pass

        if all(torch.abs(temp - theta) < tol):
            converged = True
            break

    return theta, converged


def initial_guesses(theta_ref, num_guesses, rhs):
    """theta_ref plus num_guesses x 3 random guesses (consumes torch's CPU RNG).

    Per round: theta_ref with its nonzero entries shifted by U(0.5, 1.0),
    theta_ref + U(1, 2), and U(3, 6) with the shape of rhs (n, 1).
    """
    initial_theta_list = [theta_ref]
    for _ in range(num_guesses):
        theta_trial_1 = theta_ref.clone()
        non_zero_indices = theta_ref != 0
        theta_trial_1[non_zero_indices] += torch.rand_like(theta_ref[non_zero_indices]) * 0.5 + 0.5
        theta_trial_2 = theta_ref.clone() + torch.rand_like(theta_ref) * 1 + 1
        theta_trial_3 = torch.rand_like(rhs) * 3 + 3

        initial_theta_list.append(theta_trial_1.reshape(-1))
        initial_theta_list.append(theta_trial_2.reshape(-1))
        initial_theta_list.append(theta_trial_3.reshape(-1))
    return torch.stack(initial_theta_list)


def main_Lp(model, constitutive_params, ctx, loss_history, lam_factor, num_guesses=3000, p=1 / 4, threshold=1e-3):
    """Run the sparse regression from all initial guesses; returns the best converged theta.

    lam = lam_factor * sum(FF^2).
    """
    AA, FF = generate_A_and_F(model, constitutive_params, ctx, loss_history)
    LHS, RHS = AA.T @ AA, AA.T @ FF

    theta_ref = constitutive_params.params.cpu().clone()
    initial_theta_list = initial_guesses(theta_ref, num_guesses, RHS)

    lam_i_list = torch.tensor([lam_factor]) * torch.sum(FF ** 2)

    min_residual = float('inf')
    best_theta = None
    for lam_try, lam_i in enumerate(lam_i_list):
        for i in tqdm(range(len(initial_theta_list)), desc='Lp Regression with {0}th lambda \t '.format(lam_try + 1)):
            theta_i = initial_theta_list[i]
            theta, converged = apply_Lp_threshold_regression(LHS, RHS, theta_i, p=p, lam=lam_i.item(), threshold=threshold)
            residual = compute_residual(AA, FF, theta)

            if converged and residual < min_residual:
                min_residual = residual
                best_theta = theta

    print(f"Best theta: {best_theta}")
    print(f"min_residual: {min_residual}")
    return best_theta
