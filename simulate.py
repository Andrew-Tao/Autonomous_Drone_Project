
import numpy as np

def simulator (controller, tstart, tend, dt):
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

    position_history = np.zeros(3,total_steps)
    velocity_history = np.zeros(3,total_steps)
    theta_history = np.zeros(3,total_steps)
    thetadot_history = np.zeros(3,total_steps)
    inputout_history = np.zeros(3,total_steps)

    controller_parameters = {
        'dt':dt,
        'I':I,
        'thrust_coefficient':thrust_coefficient,
        'torque_coefficient':torque_coefficient,
        'gravitational_acceleration':gravitational_acceleration,
        'mass':mass,
    }

    #Initialize the Parameters
    position = [0,0,10]
    velocity = [0,0,0]
    theta = [0,0,0]

    index = 0
    for t in ts:
        index = index + 1
        [i,controller_parameters] = controller(controller_parameters,thetadot)

        omega = thetadot2omega(thetadot, theta)
        a = compute_acceleration(i, theta, velocity, mass, gravitational_acceleration, thrust_coefficient, drag_coefficient);
        omegadot = compute_angular_acceleration(i, omega, I, length, torque_coefficient, thrust_coefficient);


        #update the parameters
        omega = omega + dt * omegadot
        thetadot = omega2thetadot(omega, theta)
        theta = theta + dt * thetadot
        velocity = velocity + dt * a
        position = position + dt * velocity



        position_history[:, i] = position
        velocity_history[:, i] = velocity
        theta_history[:, i] = theta
        thetadot_history[:, i] = thetadot
        inputout_history[:, i] = i;

        results = {
            'position': position_history,
            'theta': theta_history,
            'velocity': velocity_history,
            'angular_velocity': thetadot_history,
            'time': ts,'dt': dt, 'input': inputout_history}




def compute_thrust(inputs,thrust_coefficient):
    return [0,0,thrust_coefficient * np.sum(inputs)]

def compute_torque(inputs,length,torque_coefficient,thrust_coefficient):
    return [
        length* thrust_coefficient * (inputs[0]-inputs[2]),
        length* thrust_coefficient * (inputs[1]-inputs[3]),
        torque_coefficient * (inputs[0]-inputs[1]+inputs[2]-inputs[3]),
    ]

def compute_acceleration(
    inputs,
    angles,
    vels,
    mass,
    gravitational_acceleration,
    thrust_coefficient,
    drag_coefficient):
    gravity = [0,0,-gravitational_acceleration]
    R = rotation(angles)
    T = R * compute_torque(inputs,thrust_coefficient)
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
    omegad = np.invert(I) * (tau - np.cross (omega, I * omega) )
    return omegad


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







