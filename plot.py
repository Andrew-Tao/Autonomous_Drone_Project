import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from rotation import rotation



def plot_the_motion(position,angles):
    fig = plt.figure()


    ax = plt.axes(projection="3d")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")



    x = position[0]
    y = position[1]
    z = position[2]

    theta_x = angles[0]
    theta_y = angles[1]
    theta_z = angles[2]

    length = len(z)


#    ax.scatter(x, y, z, c = z, cmap = 'viridis')


    metadata = dict(title='Movie', artist = 'coding')
    writer = FFMpegWriter(fps=15,metadata=metadata)


    with writer.saving(fig, 'movie.mp4', 100):
        for t in range(length):


            ax.clear()
            #ax.view_init(elev=20, azim=90)
            ax.scatter(x, y, z, c=z, cmap='viridis')

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 10)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            motors_position = model_engines(position[:,t],angles[:,t])

            motor1_position = motors_position[0]
            motor2_position = motors_position[1]
            motor3_position = motors_position[2]
            motor4_position = motors_position[3]


            ax.scatter(x[t], y[t], z[t], color='blue')

            ax.scatter(motor1_position[0], motor1_position[1], motor1_position[2], color='black')
            ax.scatter(motor2_position[0], motor2_position[1], motor2_position[2], color='black')
            ax.scatter(motor3_position[0], motor3_position[1], motor3_position[2], color='black')
            ax.scatter(motor4_position[0], motor4_position[1], motor4_position[2], color='black')

            ax.plot([motor1_position[0],motor3_position[0]], [motor1_position[1],motor3_position[1]], [motor1_position[2],motor3_position[2]], color='magenta', linewidth=3, marker='o', markersize=8)
            ax.plot([motor2_position[0],motor4_position[0]], [motor2_position[1],motor4_position[1]], [motor2_position[2],motor4_position[2]], color='magenta', linewidth=3, marker='o', markersize=8)

            ax.text(motor1_position[0], motor1_position[1], motor1_position[2], 'rotor1', color='black')
            ax.text(motor2_position[0], motor2_position[1], motor2_position[2], 'rotor2', color='black')
            ax.text(motor3_position[0], motor3_position[1], motor3_position[2], 'rotor3', color='black')
            ax.text(motor4_position[0], motor4_position[1], motor4_position[2], 'rotor4', color='black')


            normal_vector = l2_normalize(rotation([theta_x[t], theta_y[t], theta_z[t]]) @ np.array([0, 0, 1]))
            dx = normal_vector[0]
            dy = normal_vector[1]
            dz = normal_vector[2]

            ax.quiver(x[t], y[t], z[t], dx, dy, dz, color='r')

            writer.grab_frame()

def l2_normalize(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def model_engines(position,angles):
    theta_x = angles[0]
    theta_y = angles[1]
    theta_z = angles[2]

    motors_position = np.zeros((4,3))
    motors_position[0] = position + (l2_normalize(rotation([theta_x, theta_y, theta_z]) @ np.array([1, 0, 0])))*0.25
    motors_position[1] = position - (l2_normalize(rotation([theta_x, theta_y, theta_z]) @ np.array([0, 1, 0])))*0.25
    motors_position[2] = position - (l2_normalize(rotation([theta_x, theta_y, theta_z]) @ np.array([1, 0, 0])))*0.25
    motors_position[3] = position + (l2_normalize(rotation([theta_x, theta_y, theta_z]) @ np.array([0, 1, 0])))*0.25

    return motors_position
