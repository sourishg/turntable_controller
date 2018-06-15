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

vae = load_model('vae_model.h5')
print('Loaded VAE model!')

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create ground plane
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)