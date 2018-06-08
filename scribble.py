import numpy as np
from keras import backend as K
import tensorflow as tf

num_rays = 100
H = 9
F = 1

num_weights = (num_rays - 2) * 4 * H + 6 * H
kernel_shape = (H * num_rays + H, num_rays)

w = K.variable(np.random.uniform(0, 1, num_weights))
w_vals = K.eval(w)

sess = K.get_session()

indices = []
idx = 0
shift = 0
k = np.zeros(shape=kernel_shape)
for i in range(num_rays):
    z = 0
    if i > 0 and i < num_rays - 1:
        z = 3*H
    else:
        z = 2*H
    for j in range(z):
        k[shift+j][i] = w_vals[idx]
        indices.append([shift+j,i])
        idx = idx + 1
    if i > 0 and i < num_rays - 1:
        shift = shift + H
    for j in reversed(range(H)):
        k[kernel_shape[0]-j-1][i] = w_vals[idx]
        indices.append([kernel_shape[0]-j-1,i])
        idx = idx + 1


#print k

for idx in indices:
    k[idx[0]][idx[1]] = 1

#print k

#print K.eval(w)
#print k

#print indices
#print w_vals

x = K.variable(np.zeros(shape=(100, kernel_shape[0])))

w1 = tf.SparseTensor(indices=indices, values=w, dense_shape=kernel_shape)
w2 = K.to_dense(w1)
w = K.variable(w2)

print K.shape(w).eval(session=sess)

cost = K.dot(x, w)
