import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from prepare_data import get_dataset
from vae_rnn import DeepVAE

TRAINED = False

FLAGS = flags.FLAGS

flags.DEFINE_integer('seq_length', 10,
                     'Length of input sequence')
flags.DEFINE_integer('pred_length', 1,
                     'Length of prediction')
flags.DEFINE_integer('num_rays', 100,
                     'Length of prediction')
flags.DEFINE_float('train_val_split', 0.8,
                   'Training/validation split ratio')

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    if not TRAINED:
        x, y, u = get_dataset(sys.argv[1])
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        train_samples = int(FLAGS.train_val_split * x.shape[0])
        train_idx = idx[:train_samples]
        val_idx = idx[train_samples:]

        x_train, x_val = x[train_idx, :], x[val_idx, :]
        y_train, y_val = y[train_idx, :], y[val_idx, :]
        u_train, u_val = u[train_idx, :], u[val_idx, :]

        print(x_train.shape, y_train.shape, u_train.shape)
        print(x_val.shape, y_val.shape, u_val.shape)

    x_test, y_test, u_test = get_dataset(sys.argv[2])

    vae = DeepVAE(FLAGS.num_rays, FLAGS.seq_length, 
                  FLAGS.pred_length, var_samples=30,
                  epochs=10, batch_size=500)

    if TRAINED:
        # load weights into new model
        vae.load_weights("vae_weights.h5")
    else:
        vae.fit(x_train, x_val,
                y_train, y_val,
                u_train, u_val)

    vae.plot_results(x_test, u_test, y_test)