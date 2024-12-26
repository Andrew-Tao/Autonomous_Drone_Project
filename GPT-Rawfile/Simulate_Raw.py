
import numpy as np
import rotation

import controller_raw

# Define constants
g = 9.81  # Gravitational acceleration (m/s^2)
m = 0.5  # Mass of quadcopter (kg)
L = 0.25  # Length of quadcopter (m)
k = 3e-6  # Thrust coefficient
b = 1e-7  # Drag coefficient
I = np.diag([5e-3, 5e-3, 10e-3])  # Inertia matrix (kg.m^2)
kd = 0.25  # Global drag coefficient


def simulate(controller=None, tstart=0, tend=10, dt=0.005):
    # Time steps
    ts = np.arange(tstart, tend, dt)
    N = len(ts)

    # Initialize output arrays
    xout = np.zeros((3, N))
    xdotout = np.zeros((3, N))
    thetaout = np.zeros((3, N))
    thetadotout = np.zeros((3, N))
    inputout = np.zeros((4, N))

    # Controller parameters
    controller_params = {
        'dt': dt,
        'I': I,
        'k': k,
        'L': L,
        'b': b,
        'm': m,
        'g': g
    }

    # Initial system state
    x = np.array([0, 0, 10])  # Position (x, y, z)
    xdot = np.zeros(3)  # Velocity (xdot, ydot, zdot)
    theta = np.zeros(3)  # Angles (roll, pitch, yaw)

    # If no controller is provided, we simulate random angular velocity.
    if controller is None:
        thetadot = np.zeros(3)
    else:
        deviation = 300  # Random deviation for angular velocity in degrees/sec
        thetadot = np.radians(2 * deviation * np.random.rand(3) - deviation)

    # Simulation loop
    for ind, t in enumerate(ts):
        # Get input from controller or default input
        if controller is None:
            i = input_function(t)
        else:
            i, controller_params = controller(controller_params, thetadot)

        # Compute forces, torques, and accelerations
        omega = thetadot_to_omega(thetadot, theta)
        a = acceleration(i, theta, xdot, m, g, k, kd)
        omegadot = angular_acceleration(i, omega, I, L, b, k)

        print(omegadot)
        # Update system state
        omega = omega + dt * omegadot
        thetadot = omega_to_thetadot(omega, theta)
        theta = theta + dt * thetadot
        xdot = xdot + dt * a
        #print(a)
        x = x + dt * xdot

        # Store simulation state
        #print (x)
        xout[:, ind] = x
        xdotout[:, ind] = xdot
        thetaout[:, ind] = theta
        thetadotout[:, ind] = thetadot
        inputout[:, ind] = i

    # Return the simulation results as a dictionary
    result = {
        'x': xout,
        'theta': thetaout,
        'vel': xdotout,
        'angvel': thetadotout,
        't': ts,
        'dt': dt,
        'input': inputout
    }

    return result


# Default input function (for testing)
def input_function(t):
    i = np.zeros(4)
    i[:] = 700
    i[0] += 150
    i[2] += 150
    i = i ** 2
    return i


# Compute thrust given current inputs and thrust coefficient
def thrust(inputs, k):
    return np.array([0, 0, k * np.sum(inputs)])


# Compute torques based on inputs, length, drag coefficient, and thrust coefficient
def torques(inputs, L, b, k):
    print("inputs", inputs)
    tau = np.array([
        L * k * (inputs[0] - inputs[2]),
        L * k * (inputs[1] - inputs[3]),
        b * (inputs[0] - inputs[1] + inputs[2] - inputs[3])
    ])
    return tau


# Compute acceleration in the inertial reference frame
def acceleration(inputs, angles, vels, m, g, k, kd):
    gravity = np.array([0, 0, -g])
    R = rotation_matrix(angles)
    T = R @ thrust(inputs, k)
    Fd = -kd * vels
    a = gravity + 1 / m * T + Fd
    return a


# Compute angular acceleration in the body frame
def angular_acceleration(inputs, omega, I, L, b, k):
    tau = torques(inputs, L, b, k)

    omegadot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega))
    return omegadot


# Convert derivatives of roll, pitch, yaw to omega
def thetadot_to_omega(thetadot, angles):
    phi, theta, psi = angles
    W = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])
    omega = W @ thetadot
    return omega


# Convert omega to roll, pitch, yaw derivatives
def omega_to_thetadot(omega, angles):
    phi, theta, psi = angles
    W = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])
    thetadot = np.linalg.inv(W) @ omega
    return thetadot


# Rotation matrix for the given Euler angles (roll, pitch, yaw)
def rotation_matrix(angles):
    phi, theta, psi = angles
    R = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])
    return R

results = simulate(controller_raw.controller('pid', 0.5, 0.01, 0.1), 0, 5, 0.01)
#print (results)

