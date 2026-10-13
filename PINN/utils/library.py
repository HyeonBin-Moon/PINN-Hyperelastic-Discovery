"""Strain-energy candidate library.

The strain-energy density is W = sum_k theta_k Q_k with 57 candidate terms Q_k:

  0-34   (I1_bar-3)^(j-i) (I2_bar-3)^i,  j = 1..7, i = 0..j  (generalized Mooney-Rivlin)
  35-41  (J-1)^(2k),                     k = 1..7
  42-48  (I2_bar^1.5 - 3^1.5)^i,         i = 1..7
  49-55  ln(J)^(2k),                     k = 1..7
  56     J - 1 - ln(J)

This module provides the analytic derivatives dQ_k/dF (first Piola-Kirchhoff
contributions), the invariants, and the ground-truth coefficients of the eight
benchmark models.
"""
import torch


NUM_TERMS = 57

# Benchmark models in the default run order.
MODEL_SYMBOLS = ['NH4', 'NH2', 'IH', 'HW', 'MR4', 'MR2', 'YH', 'HN']

# Ground-truth coefficients of each benchmark model: {library index: value}
model_theta_settings = {
    'NH2': {0: 0.5, 35: 1.5},
    'NH4': {0: 0.5, 36: 1.5},
    'IH': {0: 0.5, 1: 1.0, 2: 1.0, 35: 1.5},
    'HW': {0: 0.5, 1: 1.0, 3: 0.7, 5: 0.2, 35: 1.5},
    'MR2': {0: 0.5, 1: 1.0, 49: 1.5},
    'MR4': {0: 0.5, 1: 1.0, 50: 1.5},
    'YH': {0: 0.5, 2: 0.7, 5: 0.3, 49: 1.5},
    'HN': {0: 0.5, 43: 1.0, 56: 1.5}
}


def true_theta(model_symbol):
    """Ground-truth coefficient vector (57,) of a benchmark model."""
    if model_symbol not in model_theta_settings:
        raise ValueError(f"Unknown model symbol: {model_symbol}")
    theta = torch.zeros(NUM_TERMS)
    for index, value in model_theta_settings[model_symbol].items():
        theta[index] = value
    return theta


def compute_invariants(F):
    """J, I1_bar, I2_bar of deformation gradients F (..., 3, 3) (no clamping of J)."""
    J = torch.det(F)
    J23 = J ** (2 / 3)
    J43 = J ** (4 / 3)
    C = torch.matmul(F.transpose(-2, -1), F)
    C_dot_C = torch.einsum('...ij,...ij->...', C, C)
    I1 = torch.einsum('...ii', C)
    I1_bar = I1 / J23
    I2_bar = 0.5 * (I1_bar ** 2 - C_dot_C / J43)
    return J, I1_bar, I2_bar


def invariant_derivatives(F, J, I1_bar, I2_bar):
    """dI1_bar/dF, dI2_bar/dF and dJ/dF for F (..., 3, 3)."""
    J23 = J ** (2 / 3)
    J43 = J ** (4 / 3)
    C = torch.matmul(F.transpose(-2, -1), F)
    F_inv = torch.linalg.pinv(F)
    F_inv_T = F_inv.transpose(-2, -1).contiguous()

    dI1_bar_dF = (2 / J23[..., None, None]) * F - (2 / 3) * I1_bar[..., None, None] * F_inv_T
    dJ_dF = J[..., None, None] * F_inv_T
    dI2_bar_dF = (2 / J23[..., None, None]) * I1_bar[..., None, None] * F - (2 / J43[..., None, None]) * torch.matmul(F, C) - (4 / 3) * I2_bar[..., None, None] * F_inv_T
    return dI1_bar_dF, dI2_bar_dF, dJ_dF


def compute_analytic_dQ_dF(F):
    """Derivatives of all candidate terms w.r.t. F.

    F: (num_increments, n_elements, 3, 3) or (n_elements, 3, 3).
    Returns dQ/dF: (num_increments, n_elements, 57, 3, 3) (term axis is dim 2).
    """
    # Invariants
    J = torch.det(F)
    J = torch.clamp(J, min=1e-8, max=1e8)
    J23 = J ** (2 / 3)
    J43 = J ** (4 / 3)

    C = torch.matmul(F.transpose(-2, -1), F)
    C_dot_C = torch.einsum('...ij,...ij->...', C, C)

    I1 = torch.einsum('...ii', C)
    I1_bar = I1 / J23
    I2_bar = 0.5 * (I1_bar ** 2 - C_dot_C / J43)

    # Derivatives of the invariants
    F_inv = torch.linalg.pinv(F)
    F_inv_T = F_inv.transpose(-2, -1).contiguous()

    dI1_bar_dF = (2 / J23[..., None, None]) * F - (2 / 3) * I1_bar[..., None, None] * F_inv_T
    dJ_dF = J[..., None, None] * F_inv_T
    dI2_bar_dF = (2 / J23[..., None, None]) * I1_bar[..., None, None] * F - (2 / J43[..., None, None]) * torch.matmul(F, C) - (4 / 3) * I2_bar[..., None, None] * F_inv_T

    dQ_terms_dF = []

    # 0-34: (I1_bar-3)^(j-i) * (I2_bar-3)^i
    N = 7
    for j in range(1, N + 1):
        for i in range(j + 1):
            term1 = (j - i) * (I1_bar - 3) ** max(0, j - i - 1) * (I2_bar - 3) ** i
            term2 = i * (I2_bar - 3) ** max(0, i - 1) * (I1_bar - 3) ** (j - i)
            term1 = term1.unsqueeze(-1).unsqueeze(-1)
            term2 = term2.unsqueeze(-1).unsqueeze(-1)
            dQ_terms_dF.append(term1 * dI1_bar_dF + term2 * dI2_bar_dF)

    # 35-41: (J-1)^(2k)
    for k in range(1, N + 1):
        terms3 = 2 * k * (J - 1) ** (2 * k - 1)
        terms3 = terms3.unsqueeze(-1).unsqueeze(-1)
        dQ_terms_dF.append(terms3 * dJ_dF)

    # 42-48: (I2_bar^1.5 - 3^1.5)^i
    for i in range(1, N + 1):
        terms5 = i * ((I2_bar ** 1.5 - 3 ** 1.5) ** max(0, i - 1)) * 1.5 * (I2_bar ** 0.5)
        terms5 = terms5.unsqueeze(-1).unsqueeze(-1)
        dQ_terms_dF.append(terms5 * dI2_bar_dF)

    # 49-55: ln(J)^(2k)
    for i in range(1, N + 1):
        terms6 = 2 * i * (torch.log(J) ** max(0, 2 * i - 1)) / J
        terms6 = terms6.unsqueeze(-1).unsqueeze(-1)
        dQ_terms_dF.append(terms6 * dJ_dF)

    # 56: J - 1 - ln(J)
    term7 = 1 - 1 / J
    term7 = term7.unsqueeze(-1).unsqueeze(-1)
    dQ_terms_dF.append(term7 * dJ_dF)

    return torch.stack(dQ_terms_dF, dim=2)
