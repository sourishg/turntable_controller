from __future__ import print_function

from vae import VAE

import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

import pybullet as p
import pybullet_data

# Constants
PI = 3.14159
num_obstacles = 10
H = 8  # no of past observations
F = 4  # no of future predictions

# Control params
max_angular_velocity = 1.0  # control input sampled from [-max, max]
cur_state = 0.0  # initial state
dt = 0.1  # time increment
T = 10  # total time for one control input
M = np.linspace(-max_angular_velocity, max_angular_velocity, num=20)

# LIDAR params
lidar_pos = (0.0, 0.0, 0.2)
theta_range = np.deg2rad(120.0)
num_rays = 100  # discretization of rays
num_samples = 30 # no of variational samples
lidar_range = 5.0
theta_inc = theta_range / float(num_rays)

# Obstacle params
mass = 0

# VAE stuff
prev_ranges = []
prev_controls = []

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def norm_angle(theta):
    while theta > PI:
        theta = theta - 2 * PI
    while theta <= -PI:
        theta = theta + 2 * PI
    return theta

def create_random_world():
    # Create ground plane
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, 0)

    radius = 0.2
    max_x, max_y = 4, 4
    eps = 0.3
    x1 = np.random.uniform(-max_x, -eps, num_obstacles / 2)
    x2 = np.random.uniform(eps, max_x, num_obstacles / 2)
    x = np.concatenate((x1, x2), axis=0)
    y1 = np.random.uniform(-max_y, -eps, num_obstacles / 2)
    y2 = np.random.uniform(eps, max_y, num_obstacles / 2)
    y = np.concatenate((y1, y2), axis=0)

    obstCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2)
    UIDs = []
    for i in range(x.shape[0]):
        obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[x[i], y[i], 0],
                                    baseOrientation=[0, 0, 0, 1])
        UIDs.append(obstUID)
    return UIDs

def get_batch_ray_to(theta):
    rays = []
    theta_inc = theta_range / float(num_rays)
    for i in range(num_rays):
        ray = (lidar_pos[0] + lidar_range * math.cos(float(theta)), lidar_pos[1] + lidar_range * math.sin(float(theta)),
               lidar_pos[2])
        rays.append(ray)
        theta = norm_angle(theta + theta_inc)
    return rays

def get_range_reading(theta):
    ranges = []
    rayFroms = [lidar_pos for i in range(num_rays)]
    rayTos = get_batch_ray_to(theta)
    rayTestInfo = p.rayTestBatch(rayFroms, rayTos)
    for i in range(len(rayTestInfo)):
        # compute hit distance
        d = float(rayTestInfo[i][2]) * lidar_range
        # if no hit
        if rayTestInfo[i][0] == -1:
            d = lidar_range
        ranges.append(d)
    # Draw debug rays
    for i in range(0, len(rayTos), 5):
        p.addUserDebugLine(lidar_pos, rayTos[i], [1,0,0], lifeTime=0.6)
    return ranges

def next_state(state, u):
    return norm_angle(state + u * dt)

def init_history():
    global cur_state
    u = M[random.randint(0, M.shape[0]-1)]
    print("Initial control:", u)
    for i in range(H):
        ranges = get_range_reading(cur_state)
        cur_state = next_state(cur_state, u)
        prev_ranges.append(ranges)
        prev_controls.append(u)

def next_control(model):
    global prev_ranges, prev_controls
    x = np.asarray(prev_ranges[-H:]).flatten('F')
    distances = []
    y_pred = []
    for i in range(M.shape[0]):
        u = np.array(prev_controls[-H:])
        u = np.append(u, np.full(F, M[i]))
        u = np.tile(u, num_rays)
        y = model.predict(x, u)
        d = y[F-1][num_rays/2]
        print("u = %f, d = %f" % (M[i], d))
        distances.append(d)
        y_pred.append(y)
    idx = np.argmax(np.array(distances))
    print("Next control:", M[idx])
    return y_pred[idx], M[idx]

def simulate_controller(u):
    global cur_state, prev_ranges, prev_controls
    for i in range(F):
        ranges = get_range_reading(cur_state)
        cur_state = next_state(cur_state, u)
        prev_ranges.append(ranges)
        prev_controls.append(u)
    return prev_ranges[-F:]

if __name__ == '__main__':
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(1)

    create_random_world()
    vae = VAE(num_rays, theta_range, H, F, num_samples)
    vae.load_weights("vae_weights.h5")
    init_history()
    for i in range(10):
        y_pred, u = next_control(vae)
        y_true = simulate_controller(u)

        print("Plotting graphs...")
        fig = plt.figure()
        plots = []
        for z in range(F):
            ax = fig.add_subplot(2, F/2 + 1, z+1)
            plots.append(ax)
        for z in range(F):
            plots[z].plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], [float(u) for u in y_pred[z]], 'b.')
            plots[z].plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], [float(u) for u in y_true[z]], 'r.')
        plt.show()

    time.sleep(2)