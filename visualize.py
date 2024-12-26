import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def make_transformation_matrix(translation=None, rotation=None, scale=None):
    """Create a 4x4 transformation matrix for translation, rotation, and scaling."""
    T = np.eye(4)
    if translation is not None:
        T[:3, 3] = translation
    if rotation is not None:
        T[:3, :3] = rotation
    if scale is not None:
        S = np.diag([*scale, 1])
        T = np.dot(T, S)
    return T


def rotation_matrix(angles):
    """Generate a 3D rotation matrix."""
    rx, ry, rz = angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def draw_quadcopter(ax):
    """Draw a basic quadcopter model and return its components."""
    arms = [
        ax.plot([-5, 5], [0, 0], [0, 0], 'k-')[0],
        ax.plot([0, 0], [-5, 5], [0, 0], 'k-')[0]
    ]
    propellers = [
        ax.plot([5], [0], [0], 'bo')[0],
        ax.plot([-5], [0], [0], 'bo')[0],
        ax.plot([0], [5], [0], 'bo')[0],
        ax.plot([0], [-5], [0], 'bo')[0]
    ]
    thrusts = [
        ax.plot([5, 5], [0, 0], [0, 1], 'm-')[0],
        ax.plot([-5, -5], [0, 0], [0, 1], 'y-')[0],
        ax.plot([0, 0], [5, 5], [0, 1], 'm-')[0],
        ax.plot([0, 0], [-5, -5], [0, 1], 'y-')[0]
    ]
    return arms, propellers, thrusts


def update_quadcopter(data, t, arms, propellers, thrusts):
    """Update quadcopter position and orientation based on data."""
    dx = data['x'][:, t]
    angles = data['theta'][:, t]
    rotation = rotation_matrix(angles)
    transform = make_transformation_matrix(translation=dx, rotation=rotation)

    # Update arms and propellers
    for arm in arms:
        arm.set_data([dx[0] - 5, dx[0] + 5], [dx[1], dx[1]])
        arm.set_3d_properties([dx[2], dx[2]])
    for propeller in propellers:
        propeller.set_data([dx[0]], [dx[1]])
        propeller.set_3d_properties([dx[2]])

    # Update thrusts
    scales = np.exp(data['input'][:, t] / np.min(np.abs(data['input'][:, t])) + 5) - np.exp(6) + 1.5
    for thrust, scale in zip(thrusts, scales):
        thrust.set_data([dx[0], dx[0]], [dx[1], dx[1]])
        thrust.set_3d_properties([dx[2], dx[2] + scale])


def animate(data, arms, propellers, thrusts, ax):
    """Animate the quadcopter."""
    def update(frame):
        update_quadcopter(data, frame, arms, propellers, thrusts)
        ax.set_xlim(data['x'][0, frame] - 20, data['x'][0, frame] + 20)
        ax.set_ylim(data['x'][1, frame] - 20, data['x'][1, frame] + 20)
        ax.set_zlim(data['x'][2, frame] - 5, data['x'][2, frame] + 5)
        return arms + propellers + thrusts

    ani = FuncAnimation(plt.gcf(), update, frames=len(data['t']), interval=50, blit=False)
    plt.show()


def visualize(data):
    """Visualize the quadcopter simulation."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Quadcopter Flight Simulation')

    arms, propellers, thrusts = draw_quadcopter(ax)
    animate(data, arms, propellers, thrusts, ax)


# Example simulation data
simulation_data = {
    't': np.linspace(0, 10, 100),
    'x': np.random.rand(3, 100) * 10,
    'theta': np.random.rand(3, 100) * np.pi / 4,
    'input': np.random.rand(4, 100) * 10
}

visualize(simulation_data)
