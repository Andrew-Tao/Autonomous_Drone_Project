import numpy as np


def rotation(angles):
    """
    Compute the rotation matrix for a set of angles.

    Args:
        angles (np.array): A 3x1 vector of angles [psi, theta, phi] in radians.

    Returns:
        np.array: A 3x3 rotation matrix.
    """
    psi = angles[0]
    theta = angles[1]
    phi = angles[2]

    # Initialize rotation matrix
    R = np.zeros((3, 3))

    # First column
    R[:, 0] = [
        np.cos(phi) * np.cos(theta),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ]

    # Second column
    R[:, 1] = [
        np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi),
        np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
        np.cos(theta) * np.sin(psi)
    ]

    # Third column
    R[:, 2] = [
        np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta),
        np.cos(psi) * np.sin(phi) * np.sin(theta) - np.cos(phi) * np.sin(psi),
        np.cos(theta) * np.cos(psi)
    ]

    return R
