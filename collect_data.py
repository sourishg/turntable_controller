import pybullet as p
import pybullet_data
import math
import numpy as np
import sys
import params

# Constants
PI = 3.14159
num_obstacles = params.NUM_OBSTACLES
world_samples = params.WORLD_SAMPLES  # no of random worlds
control_samples = params.CONTROL_SAMPLES  # no of observations in each world

# Control params
max_angular_velocity = params.MAX_ANGULAR_VELOCITY  # control input sampled from [-max, max]
init_theta = 0.0  # initial state
dt = params.TIME_INCREMENT  # time increment
T = params.TOTAL_CONTROL_TIMESTEPS  # total time for one control input
M = np.linspace(-max_angular_velocity, max_angular_velocity, num=10)
# M = np.array([-1.0, 0.0, 1.0])

# LIDAR params
lidar_pos = params.LIDAR_POS
theta_range = np.deg2rad(params.LIDAR_THETA_RANGE_DEG)
num_rays = params.NUM_RAYS  # discretization of rays
lidar_range = params.LIDAR_RANGE

# Obstacle params
mass = 0

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create ground plane
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)


def normAngle(theta):
    """
    Normalize theta between (-PI, PI)
    :param theta: input angle
    :return: normalized angle
    """

    while theta > PI:
        theta = theta - 2 * PI
    while theta <= -PI:
        theta = theta + 2 * PI
    return theta


def createRandomWorld():
    """
    Generate random obstacle cylinders
    :return: set of obstacle IDs
    """

    radius = 0.2  # radius of cylinder
    max_x, max_y = 4, 4  # bounding box of obstacles
    eps = 0.3  # min distance of obstacles from robot

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


def repositionObstacles(UIDs):
    """
    Reposition obstacles
    :param UIDs: obstacle IDs
    :return:
    """

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

    i = 0
    for id in UIDs:
        p.resetBasePositionAndOrientation(id, posObj=[x[i], y[i], 0], ornObj=[0, 0, 0, 1])
        i = i + 1


def createWorld():
    """
    Create a fixed world
    :return:
    """

    obstCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3)
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2.2, 1.2, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[3, 1, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-3.5, -1, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-1, 1.5, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[0, -1.2, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[4, -2.2, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-4, 3, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[3.4, -3.4, 0], baseOrientation=[0, 0, 0, 1])
    obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-1, 1, 0], baseOrientation=[0, 0, 0, 1])


def getBatchRayTo(theta):
    """
    Generate laser ranges at a particular angle
    :param theta: current angle
    :return:
    """

    rays = []
    theta_inc = theta_range / float(num_rays)
    for i in range(num_rays):
        ray = (lidar_pos[0] + lidar_range * math.cos(float(theta)), lidar_pos[1] + lidar_range * math.sin(float(theta)),
               lidar_pos[2])
        rays.append(ray)
        theta = normAngle(theta + theta_inc)
    return rays


def getRangeReading(theta):
    """
    Simulate laser range readings
    :param theta: current angle
    :return:
    """

    ranges = []
    rayFroms = [lidar_pos for i in range(num_rays)]
    rayTos = getBatchRayTo(theta)
    rayTestInfo = p.rayTestBatch(rayFroms, rayTos)
    for i in range(len(rayTestInfo)):
        # compute hit distance
        d = float(rayTestInfo[i][2]) * lidar_range
        # if no hit
        if rayTestInfo[i][0] == -1:
            d = lidar_range
        ranges.append(d)
    # Draw debug rays
    # for i in range(len(rayTos)):
    #   p.addUserDebugLine(lidar_pos, rayTos[i], [1,0,0], lifeTime=0.1)
    return ranges


p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)

if __name__ == '__main__':
    # Open data file
    f = open(sys.argv[1], 'wb')
    # Sample worlds
    UIDs = createRandomWorld()
    for z in range(world_samples):
        print "Recording world", z
        # Sample control inputs
        # u = np.random.uniform(-max_angular_velocity, max_angular_velocity, control_samples)
        # control_idx = np.random.randint(M.shape[0], size=control_samples)
        for sample in range(control_samples):
            # data stored as: theta, theta_dot, range values...
            # for each world there are "control_samples" number of rows
            # so total rows = world_samples * control_samples * T
            # print("Recording datapoint %i\n" % i)
            u = np.random.uniform(-max_angular_velocity, max_angular_velocity)
            for j in range(T):
                ranges = getRangeReading(init_theta)
                f.write("%f %f " % (init_theta, u))
                for k in range(len(ranges)):
                    f.write("%f " % ranges[k])
                f.write("\n")
                init_theta = normAngle(init_theta + u * dt)
        # change the world
        f.write("1000\n")  # 1000 is written to file to denote end of a world
        repositionObstacles(UIDs)
