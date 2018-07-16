import sys
import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
PI = 3.14159

# LIDAR params
lidar_pos = (0.0, 0.0, 0.2)
theta_range = np.deg2rad(120.0)
num_rays = 100
lidar_range = 5.0

# Training data
state = []
ranges = []

def normAngle(theta):
  while theta > PI:
    theta = theta - 2 * PI
  while theta <= -PI:
    theta = theta + 2 * PI
  return theta

def loadData(filename):
  with open(filename) as f:
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
      parts = line.split(" ")
      state.append([float(parts[1]), float(parts[0])])
      ranges.append([float(parts[i+2]) for i in range(num_rays)])
  print("Loaded data!")

def plotMap():
  x = []
  y = []
  theta_inc = theta_range / float(num_rays)
  for i in range(len(state)):
    theta = state[i][0]
    for j in range(num_rays):
      if ranges[i][j] < lidar_range:
        x.append(lidar_pos[0] + ranges[i][j] * math.cos(theta))
        y.append(lidar_pos[1] + ranges[i][j] * math.sin(theta)) 
      theta = normAngle(theta + theta_inc)
  plt.plot(x, y, '.')
  plt.show()

def plotRange(timeStep):
  theta_inc = theta_range / float(num_rays)
  plt.plot([i * np.rad2deg(theta_inc) for i in range(num_rays)], ranges[timeStep], '.')
  plt.show()

loadData(sys.argv[1])
#plotRanges()
for i in range(20):
  plotRange(i)