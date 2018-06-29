from __future__ import print_function

import numpy as np
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def get_dataset(filename):
    x, y, u = [], [], []
    f = open(filename, "r")
    lines = [line.rstrip('\n') for line in f]
    for i in range(len(lines) - FLAGS.seq_length - FLAGS.pred_length):
        print('Read datapoint', i)
        x1, y1, u1 = [], [], []
        for j in range(FLAGS.seq_length):
            parts = lines[i + j].split(" ")
            x1.append(parts[2:FLAGS.num_rays+2:1])
            u1.append(parts[1])
        
        for j in range(FLAGS.pred_length):
            parts = lines[i + j + FLAGS.seq_length].split(" ")
            y1.append(parts[2:FLAGS.num_rays+2:1])
            u1.append(parts[1])

        x.append(x1)
        y.append(y1)
        u.append(u1)

    x = np.array(x).astype('float32')
    y = np.array(y).astype('float32')
    u = np.array(u).astype('float32')

    return x, y, u