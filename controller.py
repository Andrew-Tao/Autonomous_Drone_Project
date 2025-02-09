import numpy as np
import math

# Create a controller based on its name, using a look-up table.
def controller(name, Kd=None, Kp=None, Ki=None):
    # Use manually tuned parameters, unless arguments provide the parameters.
    if Kd is None and Kp is None and Ki is None:
        Kd = 4
        Kp = 3
        Ki = 5.5
    elif (Kd is None) != (Kp is None) or (Kd is None) != (Ki is None):
        raise ValueError('Incorrect number of parameters.')

    if name.lower() == 'pd':
        return lambda state, thetadot: pd_controller(state, thetadot, Kd, Kp)
    elif name.lower() == 'pid':
        return lambda state, thetadot: pid_controller(state, thetadot, Kd, Kp, Ki)
    elif name.lower() == 'customize':
        return lambda state, thetadot: customize_controller(state, thetadot, Kd, Kp)
    else:
        raise ValueError(f'Unknown controller type "{name}"')
    

def customize_controller(state, thetadot, Kd, Kp):
    if 'time' not in state:
        state['time'] = 0.0

    state['time'] += state['dt']

    input_signal = np.zeros(4)
    input_signal[0] = (9.81 * 0.5 / 3e-6  /4) + 1.1*1e5+math.e**(10*state['time'])
    input_signal[1] = (9.81 * 0.5 / 3e-6  /4) + 1.1*1e5+math.e**(10*state['time'])
    input_signal[2] = (9.81 * 0.5 / 3e-6  /4) + 1e5+math.e**state['time']
    input_signal[3] = (9.81 * 0.5 / 3e-6  /4) + 1e5+math.e**state['time']
    
    return input_signal, state
# Implement a PD controller. See simulate(controller).


def pd_controller(state, thetadot, Kd, Kp):
    """
    PD controller implementation.

    Args:
        state (dict): Controller state containing `integral`, `m`, `g`, `k`, `dt`.
        thetadot (np.array): Angular velocity (3x1 vector).
        Kd (float): Derivative gain.
        Kp (float): Proportional gain.

    Returns:
        tuple: (input, updated_state)
    """
    # Initialize integral to zero if it doesn't exist.
    if 'integral' not in state:
        state['integral'] = np.zeros(3)

    # Compute total thrust.
    total = (state['mass'] * state['gravitational_acceleration']) / state['thrust_coefficient'] / (
            np.cos(state['integral'][0]) * np.cos(state['integral'][1])
    )

    # Compute PD error and inputs.
    err = Kd * thetadot + Kp * state['integral']
    input_signal = err2inputs(state, err, total)

    # Update controller state.
    state['integral'] = state['integral'] + state['dt'] * thetadot

    return input_signal, state


# Implement a PID controller. See simulate(controller).
def pid_controller(state, thetadot, Kd, Kp, Ki):
    """
    PID controller implementation.

    Args:
        state (dict): Controller state containing `integral`, `integral2`, `m`, `g`, `k`, `dt`.
        thetadot (np.array): Angular velocity (3x1 vector).
        Kd (float): Derivative gain.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.

    Returns:
        tuple: (input, updated_state)
    """
    # Initialize integrals to zero if they don't exist.
    if 'integral' not in state:
        state['integral'] = np.zeros(3)
    if 'integral2' not in state:
        state['integral2'] = np.zeros(3)

    # Prevent wind-up
    if np.max(np.abs(state['integral2'])) > 0.01:
        state['integral2'].fill(0)

    # Compute total thrust.
    total = (state['mass'] * state['gravitational_acceleration']) / state['thrust_coefficient'] / (
            np.cos(state['integral'][0]) * np.cos(state['integral'][1])
    )

    # Compute error and inputs.
    err = Kd * thetadot + Kp * state['integral'] - Ki * state['integral2']
    input_signal = err2inputs(state, err, total)

    # Update controller state.
    state['integral'] = state['integral'] + state['dt'] * thetadot
    state['integral2'] = state['integral2'] + state['dt'] * state['integral']

    return input_signal, state


def err2inputs(state, err, total):
    """
    Given desired torques, desired total thrust, and physical parameters,
    solve for required system inputs.

    Args:
        state (dict): Controller state containing `I`, `k`, `L`, and `b`.
        err (np.array): Error vector (3x1) [e1, e2, e3].
        total (float): Desired total thrust.

    Returns:
        np.array: System inputs (4x1 vector).
    """
    e1 = err[0]
    e2 = err[1]
    e3 = err[2]
    Ix = state['I'][0, 0]
    Iy = state['I'][1, 1]
    Iz = state['I'][2, 2]
    k = state['thrust_coefficient']
    L = state['length']
    b = state['torque_coefficient']

    # Compute inputs
    inputs = np.zeros(4)


    inputs[0] = total / 4 - (2 * b * e1 * Ix + e3 * Iz * k * L) / (4 * b * k * L)
    inputs[1] = total / 4 + e3 * Iz / (4 * b) - (e2 * Iy) / (2 * k * L)
    inputs[2] = total / 4 - (-2 * b * e1 * Ix + e3 * Iz * k * L) / (4 * b * k * L)
    inputs[3] = total / 4 + e3 * Iz / (4 * b) + (e2 * Iy) / (2 * k * L)

    return inputs
