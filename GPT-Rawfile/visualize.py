import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def visualize_test(data):
    # Create a figure with three parts for 3D visualization, angular velocity, and displacement.
    fig = plt.figure(figsize=(10, 10))
    plots = [fig.add_subplot(3, 2, 1, projection='3d'),
             fig.add_subplot(3, 2, 2),
             fig.add_subplot(3, 2, 3)]

    # Initialize quadcopter and thrust objects
    t, thrusts = quadcopter()

    # Set axis scale and labels for 3D plot
    plots[0].set_xlim([-10, 30])
    plots[0].set_ylim([-20, 20])
    plots[0].set_zlim([5, 15])
    plots[0].set_xlabel('X Position')
    plots[0].set_ylabel('Y Position')
    plots[0].set_zlabel('Height')
    plots[0].set_title('Quadcopter Flight Simulation')

    # Start the animation
    animate(data, t, thrusts, plots)
    plt.show()


def animate(data, model, thrusts, plots):
    # Animate quadcopter using data from the simulation
    for t_idx in range(0, len(data['t']), 2):  # Skip frames for better visualization
        # The first, main part, is for the 3D visualization.
        ax = plots[0]

        # Compute translation to correct linear position coordinates
        dx = data['x'][:, t_idx]
        move = np.array([[1, 0, 0, dx[0]], [0, 1, 0, dx[1]], [0, 0, 1, dx[2]], [0, 0, 0, 1]])

        # Compute rotation (Euler angles to rotation matrix)
        angles = data['theta'][:, t_idx]
        rotate = rotation(angles)

        # Apply the translation and rotation transformations to each component of the quadcopter
        update_quadcopter_model(model, move, rotate)
        update_thrusts(thrusts, move, rotate, data['input'][:, t_idx])

        # Adjust the plot limits based on quadcopter's position
        xmin, xmax = dx[0] - 20, dx[0] + 20
        ymin, ymax = dx[1] - 20, dx[1] + 20
        zmin, zmax = dx[2] - 5, dx[2] + 5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.draw()

        # Update angular velocity and displacement plots
        multiplot(data, data['angvel'], t_idx, plots[1], 'Angular Velocity (rad/s)')
        multiplot(data, data['theta'], t_idx, plots[2], 'Angular Displacement (rad)')


def update_quadcopter_model(model, move, rotate):
    # Update the quadcopter's arms and propellers by transforming each point based on `move` and `rotate`
    for arm in model['arms']:
        arm.set_data(move[0, 3], move[1, 3])
        arm.set_3d_properties(move[2, 3])

    for prop in model['propellers']:
        prop.set_data(move[0, 3], move[1, 3])
        prop.set_3d_properties(move[2, 3])


def update_thrusts(thrusts, move, rotate, inputs):
    # Update thrusts by applying scaling and transformation
    scales = np.exp(inputs / np.min(np.abs(inputs)) + 5) - np.exp(6) + 1.5
    for i in range(4):
        s = scales[i]
        scalez = np.identity(4)
        if s < 0:
            scalez = np.dot(np.array([[np.cos(np.pi), -np.sin(np.pi), 0, 0],
                                      [np.sin(np.pi), np.cos(np.pi), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]]), np.array([[1, 0, 0, 0],
                                                                [0, 1, 0, 0],
                                                                [0, 0, np.abs(s), 0],
                                                                [0, 0, 0, 1]]))
        else:
            scalez = np.dot(np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, s, 0],
                                      [0, 0, 0, 1]]), scalez)
        thrusts[i].set_data(move[0, 3], move[1, 3])
        thrusts[i].set_3d_properties(move[2, 3])


def multiplot(data, values, ind, ax, ylabel):
    # Select the parts of the data to plot
    times = data['t'][:ind]
    values = values[:, :ind]

    # Plot in RGB, with different markers for different components
    ax.plot(times, values[0, :], 'r-', label='X')
    ax.plot(times, values[1, :], 'g.', label='Y')
    ax.plot(times, values[2, :], 'b-.', label='Z')

    # Set axes to remain constant throughout plotting
    xmin, xmax = min(data['t']), max(data['t'])
    ymin, ymax = 1.1 * np.min(values), 1.1 * np.max(values)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.legend()


def quadcopter():
    # Create quadcopter components (arms, propellers, and thrust cylinders)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create arms of the quadcopter
    arms = []
    arms.append(ax.plot([0, -5], [0, -0.25], [0, -0.25], color='black')[0])  # Arm 1
    arms.append(ax.plot([0, 5], [0, 0.25], [0, 0.25], color='black')[0])  # Arm 2
    arms.append(ax.plot([0, 0.25], [0, -5], [0, -0.25], color='black')[0])  # Arm 3
    arms.append(ax.plot([0, -0.25], [0, 5], [0, 0.25], color='black')[0])  # Arm 4

    # Create propellers at the end of each arm (represented as spheres)
    propellers = []
    propellers.append(ax.scatter(5, 5, 0, color='blue', s=50))  # Propeller 1
    propellers.append(ax.scatter(5, -5, 0, color='blue', s=50))  # Propeller 2
    propellers.append(ax.scatter(-5, 5, 0, color='blue', s=50))  # Propeller 3
    propellers.append(ax.scatter(-5, -5, 0, color='blue', s=50))  # Propeller 4

    # Thrust cylinders (represented as cylinders)
    thrusts = []
    for i in range(4):
        # Create a cylinder to represent the thrust
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(0, 1, 10)
        theta, z = np.meshgrid(theta, z)
        x = 0.1 * np.cos(theta)
        y = 0.1 * np.sin(theta)

        # Create a surface plot for the cylinder
        X = x + (5 if i % 2 == 0 else -5)
        Y = y + (5 if i < 2 else -5)
        Z = z
        thrusts.append(ax.plot_surface(X, Y, Z, color='m', alpha=0.7))

    # Return components as a dictionary
    return {'arms': arms, 'propellers': propellers}, thrusts


def rotation(angles):
    # Convert Euler angles to rotation matrix
    roll, pitch, yaw = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return np.dot(Rz, np.dot(Ry, Rx))


# Example data structure for testing
data = {
    't': np.linspace(0, 10, 100),
    'x': np.random.rand(3, 100) * 10,  # Example positions (x, y, z)
    'theta': np.random.rand(3, 100) * np.pi,  # Example angles (roll, pitch, yaw)
    'input': np.random.rand(4, 100) * 10,  # Example thrust inputs
    'angvel': np.random.rand(3, 100) * 10  # Example angular velocities
}

visualize_test(data)
