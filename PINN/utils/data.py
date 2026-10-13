"""FEM data loading and measurement noise.

Loads the Abaqus displacement and reaction-force histories, reduces them to the
2D mesh (first node layer; reaction forces of both layers are summed), and adds
Gaussian noise to the measured displacements and resultant reaction forces.
"""
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class FemData:
    """Nodal histories over 11 frames (initial state + 10 load increments).

    displacements: (11, n_nodes, 3)  [node_label, U1, U2]
    *_rf:          (11, n_edge, 3)   [node_label, RF1, RF2] of each edge
    """
    displacements: np.ndarray
    top_rf: np.ndarray
    bot_rf: np.ndarray
    right_rf: np.ndarray
    left_rf: np.ndarray


def _reaction_force_to_2d(rf):
    """Keep the first layer's node labels and sum the in-plane forces of both layers."""
    half = int(rf.shape[1] / 2)
    rf_2d = np.zeros_like(rf[:, :half, :-1])
    rf_2d[:, :, 0] = rf[:, :half, 0]
    rf_2d[:, :, 1:] = rf[:, :half, 1:-1] + rf[:, half:, 1:-1]
    return rf_2d.copy()


def load_fem_data(data_path):
    """Load displacements.npy and reaction_forces_*_LINE.npy and reduce them to 2D."""
    top_rf = np.load(os.path.join(data_path, 'reaction_forces_TOP_LINE.npy'))
    bot_rf = np.load(os.path.join(data_path, 'reaction_forces_BOT_LINE.npy'))
    right_rf = np.load(os.path.join(data_path, 'reaction_forces_RIGHT_LINE.npy'))
    left_rf = np.load(os.path.join(data_path, 'reaction_forces_LEFT_LINE.npy'))
    displacements = np.load(os.path.join(data_path, 'displacements.npy'))

    # 3D -> 2D
    displacements = displacements[:, :int(displacements.shape[1] / 2), :-1]
    data = FemData(
        displacements=displacements,
        top_rf=_reaction_force_to_2d(top_rf),
        bot_rf=_reaction_force_to_2d(bot_rf),
        right_rf=_reaction_force_to_2d(right_rf),
        left_rf=_reaction_force_to_2d(left_rf),
    )

    print("top_line_rf:", data.top_rf.shape)
    print("bot_line_rf:", data.bot_rf.shape)
    print("right_line_rf:", data.right_rf.shape)
    print("left_line_rf:", data.left_rf.shape)
    print("displacements:", data.displacements.shape)
    return data


def total_reaction_forces(data):
    """Resultant edge forces per frame: (top y, bottom y, right x, left x)."""
    return (np.sum(data.top_rf, axis=1)[:, 2].copy(),
            np.sum(data.bot_rf, axis=1)[:, 2].copy(),
            np.sum(data.right_rf, axis=1)[:, 1].copy(),
            np.sum(data.left_rf, axis=1)[:, 1].copy())


def add_noise(data, data_known_idx, noise_percent):
    """Add Gaussian noise in place (std = noise_percent * |true value|).

    Displacements: each in-plane component at the measurement nodes, increments 1-10.
    Forces: the resultant top/right force of every frame; the noise is spread
    uniformly over the edge nodes and the opposite edge gets the negative value,
    so the plate stays in equilibrium. Uses the global NumPy RNG (not seeded).
    """
    displacements = data.displacements
    for i in range(1, 11):
        for j in data_known_idx:
            for k in range(1, 3):
                standard_deviation = abs(displacements[i, j, k]).copy() * noise_percent
                displacements[i, j, k] += np.random.normal(0, standard_deviation)

    top_rf, bot_rf, right_rf, left_rf = data.top_rf, data.bot_rf, data.right_rf, data.left_rf
    for i in range(11):
        standard_deviation = abs(np.sum(top_rf[i, :, 2])).copy() * noise_percent
        noise_mag = np.random.normal(0, standard_deviation)
        top_rf[i, :, 2] += noise_mag / top_rf.shape[1]
        bot_rf[i, :, 2] -= noise_mag / bot_rf.shape[1]

        standard_deviation = abs(np.sum(right_rf[i, :, 1])).copy() * noise_percent
        noise_mag = np.random.normal(0, standard_deviation)
        right_rf[i, :, 1] += noise_mag / right_rf.shape[1]
        left_rf[i, :, 1] -= noise_mag / left_rf.shape[1]


def fit_with_constraint(data, degree):
    """Fit sum_{i=1..degree} c_i x^i (zero at x = 0) to a 1D series; returns the fitted values.

    Only used to visualise the noisy resultant forces.
    """
    def constrained_poly(x, *coeffs):
        return sum(coeffs[i] * x ** (i + 1) for i in range(degree))

    x = np.arange(len(data))
    params, _ = curve_fit(constrained_poly, x, data, p0=np.ones(degree))
    return constrained_poly(x, *params)
