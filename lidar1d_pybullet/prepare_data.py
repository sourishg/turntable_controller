from __future__ import print_function

import numpy as np
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def get_task_relevant_feature(y, half_width):
    y = np.array(y).astype('float32')
    l = FLAGS.num_rays / 2 - half_width
    u = FLAGS.num_rays / 2 + half_width
    d = np.amin(y[l:u])
    return 1.0 - d / 5.0

def get_dataset_training(filename):
    x, y, u = [], [], []
    f = open(filename, "r")
    lines = [line.rstrip('\n') for line in f]
    for i in range(len(lines) - FLAGS.seq_length - FLAGS.pred_length):
        print('Read datapoint', i)
        x1, y1, u1 = [], [], []
        for j in range(FLAGS.seq_length + FLAGS.pred_length):
            parts = lines[i + j].split(" ")
            data = [1.0 - float(parts[idx])/5.0 for idx in range(2, FLAGS.num_rays + 2, 1)]
            x1.append(data)
            if j != 0:
                y1.append(data)
            u1.append(parts[1])

        y1 = np.asarray(y1).flatten()

        x.append(x1)
        y.append(y1)
        u.append(u1)

    x = np.array(x).astype('float32')
    y = np.array(y).astype('float32')
    u = np.array(u).astype('float32')

    return x, y, u

def get_dataset_testing(filename):
    x, y, u = [], [], []
    f = open(filename, "r")
    lines = [line.rstrip('\n') for line in f]
    for i in range(len(lines) - FLAGS.seq_length - FLAGS.pred_length):
        print('Read datapoint', i)
        x1, y1, u1 = [], [], []
        for j in range(FLAGS.seq_length + FLAGS.pred_length):
            parts = lines[i + j].split(" ")
            data = [1.0 - float(parts[idx])/5.0 for idx in range(2, FLAGS.num_rays + 2, 1)]
            if j < FLAGS.seq_length:
                x1.append(data)
            else:
                if FLAGS.task_relevant:
                    y1.append(get_task_relevant_feature(parts[2:FLAGS.num_rays+2:1], 30))
                else:
                    y1.append(data)
            u1.append(parts[1])

        y1 = np.asarray(y1).flatten()

        x.append(x1)
        y.append(y1)
        u.append(u1)

    x = np.array(x).astype('float32')
    y = np.array(y).astype('float32')
    u = np.array(u).astype('float32')

    return x, y, u