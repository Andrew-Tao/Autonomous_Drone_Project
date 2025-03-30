import numpy as np
from matplotlib.pyplot import thetagrids

import rotation
import controller
import matplotlib.pyplot as plt
from plot import plot_the_motion

class Drone:
    """Quadrotor UAV physical state container"""
    
    def __init__(self, config):
        """
        Initialize drone with physical parameters
        
        Args:
            config (dict): Configuration dictionary containing:
                - mass (float): Drone mass in kg
                - I (np.array): 3x3 inertia matrix (kg·m²)
                - thrust_coefficient (float): Thrust coefficient (N·s²)
                - torque_coefficient (float): Torque coefficient (N·m·s²)
                - length (float): Arm length from center to motor (m)
        """
        # Physical constants
        self.mass = config['mass']
        self.I = config['I']
        self.thrust_coefficient = config['thrust_coefficient']
        self.torque_coefficient = config['torque_coefficient']
        self.length = config['length']
        
        # Dynamic state variables
        self.position = np.zeros(3)  # [x, y, z] in world frame (m)
        self.velocity = np.zeros(3)  # [vx, vy, vz] in world frame (m/s)
        self.theta = np.zeros(3)     # Euler angles [φ, θ, ψ] (roll, pitch, yaw) (rad)
        self.omega = np.zeros(3)     # Angular velocity in body frame (rad/s)
        self.last_inputs = np.zeros(4)  # Motor inputs [ω1², ω2², ω3², ω4²]

    @property
    def rotation_matrix(self):
        """Compute current rotation matrix from Euler angles
        
        Returns:
            np.array: 3x3 rotation matrix transforming body to world frame
        """
        return rotation(self.theta)

    def get_controller_state(self):
        """Prepare controller input state dictionary
        
        Returns:
            dict: State parameters needed by controllers
        """
        return {
            'mass': self.mass,
            'I': self.I,
            'length': self.length,
            'thrust_coefficient': self.thrust_coefficient,
            'torque_coefficient': self.torque_coefficient,
            'integral': self.theta.copy(),  # For PID controllers
            'dt': 0.01  # Default, will be overwritten in simulation
        }

def simulate (controller, tstart, tend, dt):
    # Drone configuration
    drone_config = {
        'mass': 0.5,
        'I': np.diag([5e-3, 5e-3, 10e-3]),
        'thrust_coefficient': 3e-6,
        'torque_coefficient': 1e-7,
        'length': 0.25
    }
    drone = Drone(drone_config)
    
    #Physical constants
    gravitational_acceleration  = 9.81
    drag_coefficient = 0.25

    ts = np.arange(tstart, tend, dt)
    total_steps = len(ts)


    position_history = np.zeros((3,total_steps))
    velocity_history = np.zeros((3,total_steps))
    theta_history = np.zeros((3,total_steps))
    thetadot_history = np.zeros((3,total_steps))
    inputout_history = np.zeros((4,total_steps))

    controller_params = {
        'dt':dt,
        'I':drone.I,
        'thrust_coefficient':drone.thrust_coefficient,
        'torque_coefficient':drone.torque_coefficient,
        'gravitational_acceleration':gravitational_acceleration,
        'mass':drone.mass,
        'length':drone.length,
        'time': 0.0
    }

    #Initialize the Parameters
    position = np.array([0,0,0])
   # print(position)
    velocity = np.zeros(3)
    theta = np.zeros(3)
    """
    theta[0] = 0
    theta[1] = np.pi/2
    theta[2] = 0
    """

    if controller is None:
        thetadot = np.zeros(3)
    else:
        deviation = 300  # Random deviation for angular velocity in degrees/sec
        #thetadot = np.radians(2 * deviation * np.random.rand(3) - deviation)
        thetadot=np.array([1.2 , -1, 7])

    i = -1
    for t in enumerate(ts):

        i = i + 1
        if controller is None:
            inputs = input_function(t)
        else:
            inputs, controller_params = controller(controller_params, thetadot)




        omega = thetadot2omega(thetadot, theta)

        a = compute_acceleration(inputs, theta, velocity, drone.mass, gravitational_acceleration, drone.thrust_coefficient, drag_coefficient)
        print(inputs,"thrust=",compute_thrust(inputs,drone.thrust_coefficient),"a=", a)
        omegadot = compute_angular_acceleration(inputs, omega, drone.I, drone.length, drone.torque_coefficient, drone.thrust_coefficient)


        #update the parameters
        omega = omega + dt * omegadot
        thetadot = omega2thetadot(omega, theta)
        theta = theta + dt * thetadot
        velocity = velocity + dt * a

        position = position + dt * velocity

       # print(position)


        position_history[:, i] = position
        velocity_history[:, i] = velocity
        theta_history[:, i] = theta
        thetadot_history[:, i] = thetadot
        inputout_history[:, i] = inputs

    results = {
            'position': position_history,
            'theta': theta_history,
            'velocity': velocity_history,
            'angular_velocity': thetadot_history,
            'time': ts,'dt': dt, 'input': inputout_history}
    return results


def input_function(t):
    i = np.zeros(4)
    i[:] = 700
    i[0] += 150
    i[2] += 150
    i = i ** 2
    return i

def compute_thrust(inputs,thrust_coefficient):
    return [0,0,thrust_coefficient * np.sum(inputs)]

def compute_torque(inputs,length,torque_coefficient,thrust_coefficient):

    tau = np.array([
        length* thrust_coefficient * (inputs[0]-inputs[2]),
        length* thrust_coefficient * (inputs[1]-inputs[3]),
        torque_coefficient * (inputs[0]-inputs[1]+inputs[2]-inputs[3]),
    ])
    return tau

def compute_acceleration(
    inputs,
    angles,
    vels,
    mass,
    gravitational_acceleration,
    thrust_coefficient,
    drag_coefficient):

    gravity = np.array([0,0,-gravitational_acceleration])
    R = rotation.rotation(angles)
    T = R @ compute_thrust(inputs,thrust_coefficient)
    Fd = -drag_coefficient * vels
    acceleration = gravity + ((1/mass)*T) + Fd
    return acceleration

def compute_angular_acceleration(
    inputs,
    omega,
    I,
    length,
    torque_coefficient,
    thrust_coefficient,
):

    tau = compute_torque(inputs,length,torque_coefficient,thrust_coefficient)
    omegadot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega))
    return omegadot

def thetadot2omega(thetadot, angles):

    phi, theta, psi = angles
    W = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])
    omega = np.dot(W, thetadot)
    return omega


def omega2thetadot(omega, angles):

    phi, theta, psi = angles
    W = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])
    thetadot = np.dot(np.linalg.inv(W), omega)
    return thetadot


results = simulate(controller.controller('customize', 0.5, 0.04, 0.1), 0, 3, 0.01)
position_history = results['position']
theta_history = results['theta']


time = results['time']

print(theta_history)

# Generate some random data
x = position_history[0]
y = position_history[1]
z = position_history[2]


# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c=z, cmap='viridis')  # `c` is used to color points based on their z values

# Show the plot
plt.show()




plot_the_motion(position_history,theta_history)