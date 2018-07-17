from __future__ import print_function
import sys

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.platform import flags
from prepare_data import get_dataset_training, get_dataset_testing
from model import TRFModel
from keras.models import load_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('seq_length', 5,
                     'Length of input sequence')
flags.DEFINE_integer('pred_length', 5,
                     'Length of prediction')
flags.DEFINE_integer('num_rays', 100,
                     'Length of prediction')
flags.DEFINE_float('train_val_split', 0.8,
                   'Training/validation split ratio')
flags.DEFINE_bool('task_relevant', False,
                  'Whether or not to predict task relevant features')

H = FLAGS.seq_length
F = FLAGS.pred_length
num_rays = FLAGS.num_rays
num_samples = 30

if __name__ == '__main__':
    x_test, y_test, u_test = get_dataset_testing(sys.argv[1])

    model = TRFModel(FLAGS.num_rays, FLAGS.seq_length,
                   FLAGS.pred_length, var_samples=30,
                   epochs=15, batch_size=1000)
    model.load_weights("vae_weights_p2.h5")
    model.custom_function(x_test, u_test, y_test)

