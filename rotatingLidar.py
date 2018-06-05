import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Constants
PI = 3.14159
data_samples = 2000

# Control params
max_angular_velocity = 2.0
init_theta = 0.0
dt = 0.1
T = 10

# LIDAR params
lidar_pos = (0.0, 0.0, 0.2)
theta_range = np.deg2rad(120.0)
num_rays = 100
lidar_range = 5.0

# Obstacle params
mass = 0

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def normAngle(theta):
  while theta > PI:
    theta = theta - 2 * PI
  while theta <= -PI:
    theta = theta + 2 * PI
  return theta

def createWorld():
  p.createCollisionShape(p.GEOM_PLANE)
  p.createMultiBody(0,0)
  obstCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3)
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[2.2,1.2,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[3,1,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-3.5,-1,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-1,1.5,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[0,-1.2,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[4,-2.2,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-4,3,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[3.4,-3.4,0], baseOrientation=[0,0,0,1])
  obstUID = p.createMultiBody(mass, obstCylinderId, -1, basePosition=[-1,1,0], baseOrientation=[0,0,0,1])

def getBatchRayTo(theta):
  rays = []
  theta_inc = theta_range / float(num_rays)
  for i in range(num_rays):
    ray = (lidar_pos[0] + lidar_range * math.cos(float(theta)), lidar_pos[1] + lidar_range * math.sin(float(theta)), lidar_pos[2])
    rays.append(ray)
    theta = normAngle(theta + theta_inc)
  return rays

def getRangeReading(theta):
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

createWorld()

# Sample control inputs
u = np.random.uniform(-max_angular_velocity, max_angular_velocity, data_samples)

f = open('data.txt', 'wb')

for i in range(len(u)):
  # data stored as: theta_dot, theta, range values...
  print("Recording datapoint %i\n" % i)
  for j in range(T):
    ranges = getRangeReading(init_theta)
    f.write("%f %f " % (u[i], init_theta))
    for k in range(len(ranges)):
      f.write("%f " % ranges[k])
    f.write("\n")
    init_theta = normAngle(init_theta + u[i] * dt)

p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(1)

while (1):
  keys = p.getKeyboardEvents()

time.sleep(0.01)