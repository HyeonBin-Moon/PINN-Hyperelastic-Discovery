"""Mesh utilities.

Reads the Abaqus input file (nodes, C3D6 wedge elements and edge node sets),
reduces the one-layer 3D wedge mesh to a 2D linear-triangle mesh, and computes
the element geometry (centroid Gauss points, areas and B matrices) used to
evaluate the deformation gradient from nodal displacements.
"""
import re
from dataclasses import dataclass

import numpy as np


# Node sets defined in Input_DG.inp, in the order they are read.
NODE_SET_NAMES = ('Bot_line', 'Top_line', 'Right_line', 'Left_line', 'Hole_line')


@dataclass
class Mesh:
    """2D triangle mesh with boundary node indices (all indices are 0-based).

    nodes:           (n_nodes, 3)    [node_label, x, y]
    elements:        (n_elements, 4) [element_label, n1, n2, n3] (1-based node labels)
    gaussian_points: (n_elements, 3) [element_label, x, y] of the element centroid
    element_areas:   (n_elements, 2) [element_label, area]
    B_matrices:      (n_elements, 4, 6) maps [ux1..ux3, uy1..uy3] to [F11, F22, F12, F21] - I
    """
    nodes: np.ndarray
    elements: np.ndarray
    gaussian_points: np.ndarray
    element_areas: np.ndarray
    B_matrices: np.ndarray
    top_idx: np.ndarray
    bot_idx: np.ndarray
    right_idx: np.ndarray
    left_idx: np.ndarray
    hole_idx: np.ndarray

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def node_coords(self):
        return self.nodes[:, 1:]


def _read_block(inp_data, start_keyword, end_keyword):
    """Return the lines strictly between two keyword lines of an .inp file."""
    pattern = re.compile(f'{start_keyword}(.*?){end_keyword}', re.DOTALL)
    lines = pattern.findall(inp_data)
    return lines[0].split("\n")[1:-1]


def _read_node_set(inp_data, name):
    """Read the node labels of an `*Nset, nset=<name>` block."""
    lines = _read_block(inp_data, rf'\*Nset, nset={name}', rf'\*Elset, elset={name}')
    return np.array([int(i) for i in ','.join(lines).rstrip(',').split(',')])


def read_abaqus_inp(file_path):
    """Read nodes, C3D6 elements and the edge node sets of an Abaqus input file.

    Returns:
        nodes (n, 4) float [label, x, y, z], elements (m, 7) int [label, n1..n6],
        node_sets dict {name: array of node labels}.
    """
    with open(file_path, 'r') as file:
        inp_data = file.read()

    lines = _read_block(inp_data, r'\*Node', r'\*Element, type=C3D6')
    nodes = np.array([list(map(float, i.split(','))) for i in lines])

    lines = _read_block(inp_data, r'\*Element, type=C3D6', r'\*Nset, nset=Set-1, generate')
    elements = np.array([list(map(int, i.split(','))) for i in lines])

    node_sets = {name: _read_node_set(inp_data, name) for name in NODE_SET_NAMES}
    return nodes, elements, node_sets


def reduce_to_2d(nodes, elements, node_sets):
    """Keep the first node layer of the one-layer wedge mesh (3D -> 2D).

    The first half of the nodes is the bottom layer; the first three nodes of
    each C3D6 element form the bottom triangle.
    """
    nodes = nodes[:int(nodes.shape[0] / 2), :-1]
    elements = elements[:, :-3]
    node_sets = {name: s[:int(len(s) / 2)] for name, s in node_sets.items()}
    return nodes, elements, node_sets


def calculate_gaussian_points_areas_and_B_matrices(nodes, elements):
    """Centroid Gauss points, areas and B matrices of linear triangles."""
    element_node_coords = nodes[:, 1:][elements[:, 1:] - 1]  # (n_elements, 3, 2)

    # Gauss integration point (triangle centroid)
    gauss_point = np.array([1.0 / 3.0, 1.0 / 3.0])  # (xi, eta)
    N = np.array([
        1.0 - gauss_point[0] - gauss_point[1],  # N1
        gauss_point[0],                         # N2
        gauss_point[1]                          # N3
    ])
    gaussian_points_coords = np.einsum('j,ijk->ik', N, element_node_coords)

    # Element areas
    x1, y1 = element_node_coords[:, 0, 0], element_node_coords[:, 0, 1]
    x2, y2 = element_node_coords[:, 1, 0], element_node_coords[:, 1, 1]
    x3, y3 = element_node_coords[:, 2, 0], element_node_coords[:, 2, 1]
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    # Shape-function derivatives
    dN1_dx = (y2 - y3) / (2 * areas)
    dN2_dx = (y3 - y1) / (2 * areas)
    dN3_dx = (y1 - y2) / (2 * areas)
    dN1_dy = (x3 - x2) / (2 * areas)
    dN2_dy = (x1 - x3) / (2 * areas)
    dN3_dy = (x2 - x1) / (2 * areas)

    # B matrix: rows [F11, F22, F12, F21], columns [ux1, ux2, ux3, uy1, uy2, uy3]
    B_matrices = np.zeros((elements.shape[0], 4, 6))
    B_matrices[:, 0, 0] = dN1_dx  # F11
    B_matrices[:, 0, 1] = dN2_dx
    B_matrices[:, 0, 2] = dN3_dx
    B_matrices[:, 1, 3] = dN1_dy  # F22
    B_matrices[:, 1, 4] = dN2_dy
    B_matrices[:, 1, 5] = dN3_dy
    B_matrices[:, 2, 0] = dN1_dy  # F12
    B_matrices[:, 2, 1] = dN2_dy
    B_matrices[:, 2, 2] = dN3_dy
    B_matrices[:, 3, 3] = dN1_dx  # F21
    B_matrices[:, 3, 4] = dN2_dx
    B_matrices[:, 3, 5] = dN3_dx

    # Prepend the element labels
    gaussian_points = np.hstack((elements[:, [0]], gaussian_points_coords))
    element_areas = np.hstack((elements[:, [0]], areas[:, np.newaxis]))

    return gaussian_points, element_areas, B_matrices


def load_mesh(inp_path):
    """Read an Abaqus .inp file and build the 2D triangle mesh."""
    nodes, elements, node_sets = read_abaqus_inp(inp_path)
    nodes, elements, node_sets = reduce_to_2d(nodes, elements, node_sets)

    print("nodes:", nodes.shape)
    print("elements:", elements.shape)
    for name in NODE_SET_NAMES:
        print(f"{name}_nodes:", node_sets[name].shape)

    gaussian_points, element_areas, B_matrices = calculate_gaussian_points_areas_and_B_matrices(nodes, elements)
    print("gaussian_points:", gaussian_points.shape)
    print("element_areas:", element_areas.shape)
    print("B_matrices:", B_matrices.shape)

    return Mesh(
        nodes=nodes,
        elements=elements,
        gaussian_points=gaussian_points,
        element_areas=element_areas,
        B_matrices=B_matrices,
        top_idx=node_sets['Top_line'] - 1,
        bot_idx=node_sets['Bot_line'] - 1,
        right_idx=node_sets['Right_line'] - 1,
        left_idx=node_sets['Left_line'] - 1,
        hole_idx=node_sets['Hole_line'] - 1,
    )


def select_closest_points(node_coords, reference_points):
    """Index of the node closest to each reference point."""
    selected_indices = []
    for ref_point in reference_points:
        distances = np.linalg.norm(node_coords - ref_point, axis=1)
        selected_indices.append(np.argmin(distances))
    return np.array(selected_indices)


def select_measurement_nodes(node_coords, hole_radius, angles_deg):
    """Nodes on the hole edge at the given angles (the displacement measurement points)."""
    reference_points = np.array([
        [hole_radius * np.cos(np.deg2rad(a)), hole_radius * np.sin(np.deg2rad(a))] for a in angles_deg
    ])
    return select_closest_points(node_coords, reference_points)
