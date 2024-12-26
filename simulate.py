import numpy as np
import rotation
import controller
import matplotlib.pyplot as plt

def simulate (controller, tstart, tend, dt):
    #Physical constants
    gravitational_acceleration  = 9.81
    mass = 0.5
    length = 0.25
    thrust_coefficient = 3e-6
    torque_coefficient = 1e-7
    I = np.zeros((3,3))
    np.fill_diagonal(I, [5e-3,5e-3,10e-3])
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
        'I':I,
        'thrust_coefficient':thrust_coefficient,
        'torque_coefficient':torque_coefficient,
        'gravitational_acceleration':gravitational_acceleration,
        'mass':mass,
        'length':length
    }

    #Initialize the Parameters
    position = np.array([0,0,10])
   # print(position)
    velocity = np.zeros(3)
    theta = np.zeros(3)

    if controller is None:
        thetadot = np.zeros(3)
    else:
        deviation = 300  # Random deviation for angular velocity in degrees/sec
        #thetadot = np.radians(2 * deviation * np.random.rand(3) - deviation)
        thetadot=np.array([-0.90556813 , - 2.41338998 , 5.17105039])

    i = -1
    for t in enumerate(ts):
        i = i + 1
        if controller is None:
            inputs = input_function(t)
        else:
            inputs, controller_params = controller(controller_params, thetadot)

       # print(inputs)

        omega = thetadot2omega(thetadot, theta)

        a = compute_acceleration(inputs, theta, velocity, mass, gravitational_acceleration, thrust_coefficient, drag_coefficient)
      #  print("a=", a)
        omegadot = compute_angular_acceleration(inputs, omega, I, length, torque_coefficient, thrust_coefficient)


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


results = simulate(controller.controller('pid', 0.5, 0.04, 0.1), 0, 5, 0.01)

position_history = results['position']
time = results['time']

plt.plot(time, position_history[0], color='black', marker='*',linestyle='--', label='Position_x')
plt.plot(time, position_history[1], color='red', marker='*',linestyle=':', label='Position_y')
plt.plot(time, position_history[2], color='blue', marker='*',linestyle=':', label='Position_z')

plt.title('Drone_Position')
      # Title
plt.xlabel('time')                     # X-axis label
plt.ylabel('x_y_z position')                     # Y-axis label
plt.grid()
plt.legend()# Optional: Add a grid
plt.show()                                    # Display the plot


