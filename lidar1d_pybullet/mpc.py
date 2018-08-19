from __future__ import print_function

from model import TRFModel
from prepare_data import get_task_relevant_feature

import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

import pybullet as p
import pybullet_data
import params

FLAGS = params.FLAGS

# Constants
PI = 3.14159
num_obstacles = params.NUM_OBSTACLES
H = FLAGS.seq_length  # no of past observations
F = FLAGS.pred_length  # no of future predictions
num_samples = params.VARIATIONAL_SAMPLES  # no of variational samples

# Control params
max_angular_velocity = params.MAX_ANGULAR_VELOCITY  # control input sampled from [-max, max]
cur_state = -1.047  # initial state
dt = params.TIME_INCREMENT  # time increment
T = params.TOTAL_CONTROL_TIMESTEPS  # total time for one control input
M = np.linspace(-max_angular_velocity, max_angular_velocity, num=10)

# LIDAR params
lidar_pos = params.LIDAR_POS
theta_range = np.deg2rad(params.LIDAR_THETA_RANGE_DEG)
num_rays = FLAGS.num_rays  # discretization of rays
lidar_range = params.LIDAR_RANGE
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

    obstCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2)
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

    obstCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius)
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
    print("Initial control:", u)
    for i in range(H):
        ranges = get_range_reading(cur_state)
        cur_state = next_state(cur_state, u)
        prev_ranges.append(ranges)
        prev_controls.append(u)


def get_risk_metric(y):
    return y[F-1]


def get_desampled_y(sampled_y):
    y = np.array(sampled_y).astype('float32')
    return np.mean(y, axis=0)


def next_control(models):
    global prev_ranges, prev_controls
    x = np.array(prev_ranges[-H:]).astype('float32')
    distances = []
    y_pred = []
    for i in range(M.shape[0]):
        u = np.array(prev_controls[-H:])
        u = np.append(u, np.full(F, M[i])).astype('float32')
        sampled_y = predict(x, u, models)
        y = get_desampled_y(sampled_y)
        d = get_risk_metric(y)

        print("u = %f, d = %f" % (M[i], d))
        distances.append(d)
        y_pred.append(sampled_y)
    idx = np.argmin(np.array(distances))
    print("Next control:", M[idx])
    return y_pred[idx], M[idx]


def simulate_controller(u):
    global cur_state, prev_ranges, prev_controls
    d = 0.0
    for i in range(F):
        cur_state = next_state(cur_state, u)
        ranges = get_range_reading(cur_state)
        prev_ranges.append(ranges)
        prev_controls.append(u)
        d = get_task_relevant_feature(ranges, FLAGS.tr_half_width)
        print("Central dist:", d)
    return d, prev_ranges[-F:]


def predict(x, control, model):
    encoder = model.get_encoder_model()
    transition_model = model.get_transition_model()
    cost_model = model.get_cost_model()

    init_outputs = []
    prev_x = x[-1]
    for p in range(F):
        z = np.random.standard_normal(model.latent_dim)
        u = control[H - 1 + p]
        y_pred = gen_model.predict([np.array([prev_y, ]), np.array([u, ]), np.array([z, ])], batch_size=1)[0]
        init_outputs.append(y_pred)
        prev_y = y_pred

    init_outputs = np.array(init_outputs).astype('float32')
    prev_x = np.array(x).astype('float32')
    x_enc = np.concatenate((prev_x, init_outputs))
    sampled_y = []
    for i in range(num_samples):
        y_pred = vae_model_tr.predict([np.array([x_enc, ]), np.array([control, ])], batch_size=1)[0]
        y = y_pred[H - 1:]
        sampled_y.append(y)
    return sampled_y

def solve_mpc(prev_x):
    """
    for i in range(30):
        _, _, z0 = encoder.predict(np.array([prev_x, ]), batch_size=1)
        u0 = [0.5]
        cost = cost_model.predict([np.array([z0[0], ]), np.array([u0, ])], batch_size=1)[0]
        cost2 = np.matmul(z0, np.matmul(U, z0.transpose())) + np.matmul(np.array([u0, ]), np.matmul(V, np.array([u0, ]).transpose()))
        print(cost, cost2)
    """
    _, _, z0 = encoder.predict(np.array([prev_x, ]), batch_size=1)
    T = FLAGS.pred_length
    z = Variable(model.latent_dim, T + 1)
    u = Variable(model.control_dim, T)

    states = []
    for t in range(T):
        if t == T-1:
            cost = quad_form(z[:, t+1], U) + quad_form(u[:, t], V)
        else:
            cost = 0
        # cost = sum_squares(z[:, t + 1]) + sum_squares(u[:, t])
        constr = [z[:, t + 1] == A * z[:, t] + B * u[:, t]]
        states.append(Problem(Minimize(cost), constr))
    # sums problem objectives and concatenates constraints.
    prob = sum(states)
    prob.constraints += [z[:, 0] == z0.transpose()]
    prob.constraints += [u <= 1.0]
    prob.constraints += [u >= -1.0]
    prob.solve(verbose=True)
    print(u.value)


model = TRFModel(FLAGS.num_rays, FLAGS.seq_length,
                 FLAGS.pred_length, var_samples=num_samples,
                 task_relevant=FLAGS.task_relevant)

if FLAGS.task_relevant:
    model.load_weights("vae_weights_tr_p2.h5")
else:
    model.load_weights("vae_weights_p2.h5")

encoder = model.get_encoder_model()
transition_model = model.get_transition_model()
cost_model = model.get_cost_model()

A = transition_model.layers[2].get_weights()[0].transpose()
B = transition_model.layers[3].get_weights()[0].transpose()

P = cost_model.layers[2].get_weights()[0]
Q = cost_model.layers[3].get_weights()[0]

U = np.matmul(P, P.transpose())
V = np.matmul(Q, Q.transpose())

print(A.shape, B.shape, P.shape, U.shape, Q.shape, V.shape)

if __name__ == '__main__':
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(1)

    create_random_world()
    # create_world()
    init_history()

    solve_mpc(prev_ranges[-1])

    """
    d = 1.0
    min_thresh = 4.5
    while d > 1.0 - 5.5/5.0:
        y_pred, u = next_control(model)
        d, prev_gt_ranges = simulate_controller(u)
        print("Current d/ Predicted d:", d, y_pred[F-1])

        print("Plotting graphs...")
        y_true = []
        for y in prev_gt_ranges:
            y_true.append(get_task_relevant_feature(y, FLAGS.tr_half_width))
        fig = plt.figure()
        plt.ylim((0.0, 1.0))
        for z in range(num_samples):
            plt.plot([j for j in range(F)], [float(u) for u in y_pred[z]], 'b.')
        plt.plot([j for j in range(F)], [float(u) for u in y_true], 'r.')
        plt.show()
    """
    while True:
        x = 1
    time.sleep(2)
