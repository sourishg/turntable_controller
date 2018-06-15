from __future__ import print_function

from keras.models import load_model

import tensorflow as tf

import time
import math
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
H = 6  # no of past observations
F = 4  # no of future predictions

# Control params
max_angular_velocity = 1.0  # control input sampled from [-max, max]
init_theta = 0.0  # initial state
dt = 0.1  # time increment
T = 10  # total time for one control input
M = np.linspace(-max_angular_velocity, max_angular_velocity, num=20)

# LIDAR params
lidar_pos = (0.0, 0.0, 0.2)
theta_range = np.deg2rad(120.0)
num_rays = 100  # discretization of rays
lidar_range = 5.0

# Obstacle params
mass = 0

# VAE stuff
prev_ranges = []
prev_controls = []

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create ground plane
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)

def vae_loss_function(y_true, y_pred):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    mse_loss = mse(y_true, y_pred)
    mse_loss *= original_dim
    return mse_loss + kl_loss

def norm_angle(theta):
    while theta > PI:
        theta = theta - 2 * PI
    while theta <= -PI:
        theta = theta + 2 * PI
    return theta

def create_random_world():
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
        theta = normAngle(theta + theta_inc)
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
    for i in range(len(rayTos)):
        p.addUserDebugLine(lidar_pos, rayTos[i], [1,0,0], lifeTime=0.1)
    return ranges

def init_history():
    for i in range(H):
        ranges = get_range_reading(init_theta)
        prev_ranges.append(ranges)

vae = load_model('vae_model.h5')
print('Loaded VAE model!')

create_random_world()
init_history()