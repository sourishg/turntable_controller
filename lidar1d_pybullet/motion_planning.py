from __future__ import print_function

from model import TRFModel

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

from tensorflow.python.platform import flags

TRAINED = True

FLAGS = flags.FLAGS

flags.DEFINE_integer('seq_length', 5,
                     'Length of input sequence')
flags.DEFINE_integer('pred_length', 5,
                     'Length of prediction')
flags.DEFINE_integer('num_rays', 100,
                     'Length of prediction')
flags.DEFINE_float('train_val_split', 0.8,
                   'Training/validation split ratio')
flags.DEFINE_bool('task_relevant', True,
                  'Whether or not to predict task relevant features')

# Constants
PI = 3.14159
num_obstacles = 10
H = FLAGS.seq_length  # no of past observations
F = FLAGS.pred_length  # no of future predictions

# Control params
max_angular_velocity = 1.0  # control input sampled from [-max, max]
cur_state = -1.047  # initial state
dt = 0.1  # time increment
T = 10  # total time for one control input
M = np.linspace(-max_angular_velocity, max_angular_velocity, num=10)

# LIDAR params
lidar_pos = (0.0, 0.0, 0.2)
theta_range = np.deg2rad(120.0)
num_rays = FLAGS.num_rays  # discretization of rays
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

def create_world():
    # Create ground plane
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, 0)

    obstCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3)
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2.2, -2, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2.3, -1.5, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2.4, -1.0, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[1.9, -0.5, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2, 0, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2.5, 0.5, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2, 1.0, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2, 1.5, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2, 2, 0], baseOrientation=[0, 0, 0, 1])

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

    np.random.shuffle(x)
    np.random.shuffle(y)

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
        ranges.append(1 - d/lidar_range)
    # Draw debug rays
    for i in range(0, len(rayTos), 5):
        p.addUserDebugLine(lidar_pos, rayTos[i], [1,0,0], lifeTime=0.6)
    return ranges

def next_state(state, u):
    return norm_angle(state + u * dt)

def init_history():
    global cur_state
    u = 0
    #u = M[random.randint(0, M.shape[0]-1)]
    print("Initial control:", u)
    for i in range(H):
        ranges = get_range_reading(cur_state)
        cur_state = next_state(cur_state, u)
        prev_ranges.append(ranges)
        prev_controls.append(u)

def get_tr_features(ranges, half_width):
    ret = []
    l = FLAGS.num_rays / 2 - half_width
    u = FLAGS.num_rays / 2 + half_width
    for r in ranges:
        d = np.amax(r[l:u])
        ret.append(d)
    return np.array(ret).astype('float32')

def get_risk_metric(y):
    return y[F-1][0]

def next_control(model):
    global prev_ranges, prev_controls
    x = np.array(prev_ranges[-H:]).astype('float32')
    distances = []
    y_pred = []
    for i in range(M.shape[0]):
        u = np.array(prev_controls[-H:])
        u = np.append(u, np.full(F, M[i])).astype('float32')
        y = model.predict(x, u)
        
        #d = y[F-1][num_rays/2]
        #d = np.amin(y[F-1][y[F-1] > 0])
        d = get_risk_metric(y)

        print("u = %f, d = %f" % (M[i], d))
        distances.append(d)
        y_pred.append(y)
    idx = np.argmin(np.array(distances))
    print("Next control:", M[idx])
    return y_pred[idx], M[idx]

def simulate_controller(u):
    global cur_state, prev_ranges, prev_controls
    d = 0.0
    for i in range(1):
        ranges = get_range_reading(cur_state)
        cur_state = next_state(cur_state, u)
        prev_ranges.append(ranges)
        prev_controls.append(u)
        d = get_tr_features([ranges], 30)[0]
        print("Central dist:", d)
    return d, prev_ranges[-F:]

if __name__ == '__main__':
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(1)

    create_random_world()
    # create_world()
    vae = TRFModel(num_rays, H, F, num_samples)
    vae.load_weights("vae_weights_tr.h5")
    init_history()
    d = 1.0
    min_thresh = 4.5
    while d > 1.0 - 4.5/5.0:
        y_pred, u = next_control(vae)
        d, prev_gt_ranges = simulate_controller(u)
        y_true = get_tr_features(prev_gt_ranges, 30)
        print("Current d/ Predicted d:", d, y_pred[F-1])
        '''
        print("Plotting graphs...")
        fig = plt.figure()
        plt.ylim((0.0, 1.0))
        for z in range(F):
            plt.plot([j for j in range(F)], [float(u) for u in y_pred], 'b.')
            plt.plot([j for j in range(F)], [float(u) for u in y_true], 'r.')
        plt.show()
        '''
    time.sleep(2)