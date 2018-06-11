from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Conv1D, UpSampling1D, Flatten
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import LambdaCallback
from keras import initializers, activations

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Constants and params
theta_range = np.deg2rad(120.0)
num_rays = 100
trained = False

H = 8  # no of past observations
F = 2  # no of future predictions

training_data_fraction = 0.8
tf_session = K.get_session()

def prepareDatasetOld(train_file, test_file):
    x_train, x_val, y_train, y_val, x_test, y_test = [], [], [], [], [], []
      
    if not trained:
        f1 = open(train_file, "r")
        x_raw = []
        y_raw = []
        lines = [line.rstrip('\n') for line in f1]
        for i in range(len(lines)-H-F):
            print("preparing training data point", i)
            x = []
            y = []
            u = []
            for j in range(H):
                parts = lines[i+j].split(" ")
                for k in range(num_rays):
                    x.append(float(parts[k+2]))
                u.append(float(parts[0]))
            for j in range(F):
                parts = lines[i+j+H].split(" ")
                for k in range(num_rays):
                    y.append(float(parts[k+2]))
            x = x + u
            x_raw.append(x)
            y_raw.append(y)

        x = np.asarray(x_raw)
        y = np.asarray(y_raw)

        n = len(lines)-H-F
        n_train_samples = int(training_data_fraction * n)
        n_val_samples = n - n_train_samples

        training_idx = np.random.randint(x.shape[0], size=n_train_samples)
        val_idx = np.random.randint(x.shape[0], size=n_val_samples)
        
        x_train, x_val = x[training_idx,:], x[val_idx,:]
        y_train, y_val = y[training_idx,:], y[val_idx,:]

        print("Prepared training dataset!")

    f2 = open(test_file, "r")
    x_raw = []
    y_raw = []
    lines = [line.rstrip('\n') for line in f2]
    for i in range(len(lines)-H-F):
        print("preparing testing data point", i)
        x = []
        y = []
        u = []
        for j in range(H):
            parts = lines[i+j].split(" ")
            for k in range(num_rays):
                x.append(float(parts[k+2]))
            u.append(float(parts[0]))
        for j in range(F):
            parts = lines[i+j+H].split(" ")
            for k in range(num_rays):
                y.append(float(parts[k+2]))
        x = x + u
        x_raw.append(x)
        y_raw.append(y)

    x_test = np.asarray(x_raw)
    y_test = np.asarray(y_raw)    

    return x_train, y_train, x_val, y_val, x_test, y_test

def prepareDataset(train_file, test_file):
    x_train, x_val, y_train, y_val, x_test, y_test = [], [], [], [], [], []

    if not trained:
        f1 = open(train_file, "r")
        x_raw = []
        y_raw = []
        lines = [line.rstrip('\n') for line in f1]
        for i in range(len(lines) - H - F):
            print("preparing training data point", i)
            x = []
            y = []
            u = []
            for j in reversed(range(H)):
                parts = lines[i + j].split(" ")
                #for k in range(num_rays):
                #    x.append(float(parts[k + 2]))
                x.append(parts[2:num_rays+2:1])
                u.append(float(parts[0]))
            for j in range(F):
                parts = lines[i + j + H].split(" ")
                #for k in range(num_rays):
                #    y.append(float(parts[k + 2]))
                y.append(parts[2:num_rays+2:1])
            
            x = np.asarray(x)
            u = np.asarray([u])
            x = np.concatenate((x, u.T), axis=1)
            x = x.flatten('F')

            y = np.asarray(y)
            y = y.flatten('F')

            x_raw.append(x)
            y_raw.append(y)

        x = np.asarray(x_raw)
        y = np.asarray(y_raw)

        n = len(lines) - H - F
        n_train_samples = int(training_data_fraction * n)
        n_val_samples = n - n_train_samples

        training_idx = np.random.randint(x.shape[0], size=n_train_samples)
        val_idx = np.random.randint(x.shape[0], size=n_val_samples)

        x_train, x_val = x[training_idx, :], x[val_idx, :]
        y_train, y_val = y[training_idx, :], y[val_idx, :]

        print("Prepared training dataset!")

    f2 = open(test_file, "r")
    x_raw = []
    y_raw = []
    lines = [line.rstrip('\n') for line in f2]
    for i in range(len(lines) - H - F):
        print("preparing testing data point", i)
        x = []
        y = []
        u = []
        for j in reversed(range(H)):
            parts = lines[i + j].split(" ")
            #for k in range(num_rays):
            #    x.append(float(parts[k + 2]))
            x.append(parts[2:num_rays+2:1])
            u.append(float(parts[0]))
        for j in range(F):
            parts = lines[i + j + H].split(" ")
            #for k in range(num_rays):
            #    y.append(float(parts[k + 2]))
            y.append(parts[2:num_rays+2:1])

        x = np.asarray(x)
        u = np.asarray([u])
        x = np.concatenate((x, u.T), axis=1)
        x = x.flatten('F')

        y = np.asarray(y)
        y = y.flatten('F')
        
        x_raw.append(x)
        y_raw.append(y)

    x_test = np.asarray(x_raw)
    y_test = np.asarray(y_raw)

    print("Prepared testing dataset!")

    return x_train, y_train, x_val, y_val, x_test, y_test


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
      args (tensor): mean and log of variance of Q(z|X)
    # Returns:
      z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def unpack_output(y):
    return np.reshape(y, (F, num_rays), order='F')

def plot_results(models,
                 data,
                 batch_size=128,
                 num_samples=1):
    # encoder, decoder = models
    encoder, decoder = models
    x, y = data
    theta_inc = theta_range / float(num_rays)
    for k in range(120, 300, 1):
        fig = plt.figure()
        y1 = unpack_output(y[k])
        plots = []
        for p in range(1, F+1):
            ax = fig.add_subplot(1, 2, p)
            plots.append(ax)
        
        for i in range(num_samples):
            _, _, z = encoder.predict(np.array([x[k], ]), batch_size=batch_size)
            y_pred = decoder.predict(z, batch_size=batch_size)
            #print(y_pred.shape)
            y_pred = unpack_output(y_pred[0])
            for p in range(1, F+1):
                plots[p-1].plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], [float(u) for u in y_pred[p-1]], 'b.')
            #plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y_pred[0], 'b.')
            #print("ypred:", y_pred[0])
        
        for p in range(1, F+1):
            plots[p-1].plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], [float(u) for u in y1[p-1]], 'r.')
        #plt.ylabel("output")

        '''
        plt.figure()
        x_plot = np.asarray(np.split(x[k][:(H*num_rays)], H))
        for i in range(H):
          plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], x_plot[i][:num_rays], 'b.')
        plt.ylabel("input")
        '''

        plt.show()


x_train, y_train, x_val, y_val, x_test, y_test = prepareDataset(sys.argv[1], sys.argv[2])
#x_train, y_train, x_val, y_val, x_test, y_test = prepareDatasetOld(sys.argv[1], sys.argv[2])
#print(x_train.shape, x_test.shape, x_val.shape, y_train.shape)

#x_train = np.expand_dims(x_train, axis=2)
#x_val = np.expand_dims(x_val, axis=2)
#x_test = np.expand_dims(x_test, axis=2)

# network parameters
original_dim = x_test.shape[1]
output_dim = y_test.shape[1]

conv1_filters = 10
dim2 = 505
batch_size = 128
latent_dim = 50
epochs = 5
num_samples = 30
input_shape = (original_dim,)
output_shape = (output_dim,)

class RotateLIDAR(Layer):

    def __init__(self, num_rays, 
                 num_past_obs,
                 num_future_pred,
                 activation,
                 **kwargs):
        self.num_rays = num_rays
        self.H = num_past_obs
        self.F = num_future_pred
        self.kernel_shape = (self.H * self.num_rays + self.H, self.num_rays)
        self.output_dim = num_future_pred * num_rays
        self.activation = activations.get(activation)

        self.w_inits = np.random.uniform(-1, 1, ((self.num_rays - 2) * 4 * self.H + 6 * self.H)).astype('float32')
        self.ids = []
        idx = 0
        shift = 0
        for i in range(self.num_rays):
            z = 0
            if i > 0 and i < self.num_rays - 1:
                z = 3*self.H
            else:
                z = 2*self.H
            for j in range(z):
                self.ids.append([shift+j,i])
                idx = idx + 1
            if i > 0 and i < self.num_rays - 1:
                shift = shift + self.H
            for j in reversed(range(self.H)):
                self.ids.append([self.kernel_shape[0]-j-1,i])
                idx = idx + 1

        super(RotateLIDAR, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        '''
        self.kernel2 = self.add_weight(name='kernel2', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        '''
        k1 = tf.SparseTensor(indices=self.ids, values=self.w_inits, dense_shape=self.kernel_shape)
        #print K.shape(k1).eval(session=tf_session)
        k2 = tf.sparse_tensor_to_dense(k1, validate_indices=False)
        #print K.shape(k2).eval(session=tf_session)
        #print K.shape(temp).eval(session=tf_session)
        
        self.w = K.variable(k2)
        #self.w = K.variable(self.w_inits)
        self.trainable_weights = [self.w]

        super(RotateLIDAR, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #w = tf.SparseTensor(indices=self.ids, values=self.w, dense_shape=self.kernel_shape)
        output = K.dot(x, self.w)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(original_dim,), name='encoder_input')
#x1 = RotateLIDAR(num_rays, H, F, activation='relu')(inputs)
x11 = Reshape((original_dim, 1), input_shape=(original_dim,))(inputs)
#x11 = Reshape((num_rays, 1), input_shape=(num_rays,))(x1)
#x1 = Conv1D(conv1_filters, 9, strides=9, activation='relu', input_shape=(None, original_dim))(inputs)
#xf = Flatten()(x1)
#x2 = Dense(dim2, activation='relu')(x1)
x2 = Conv1D(conv1_filters, H, strides=H, activation='relu', input_shape=(None, original_dim, 1))(x11)
#x2 = Conv1D(conv1_filters, 3, activation='relu', input_shape=(None, original_dim, 1))(x11)
xf = Flatten()(x2)
x3 = Dense(dim2, activation='relu')(xf)
x4 = Dense(200, activation='relu')(x3)
z_mean = Dense(latent_dim, name='z_mean')(x4)
z_log_var = Dense(latent_dim, name='z_log_var')(x4)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
y2 = Dense(dim2, activation='relu')(latent_inputs)
outputs = Dense(output_dim, activation='relu')(y2)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(encoder.layers[1].get_weights()))

def vae_loss_function(y_true, y_pred):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    mse_loss = mse(y_true, y_pred)
    mse_loss *= original_dim
    return K.mean(mse_loss + kl_loss)


if __name__ == '__main__':
    models = (encoder, decoder)
    test_data = (x_test, y_test)

    if trained:
        # load weights into new model
        vae.load_weights("vae_weights.h5")
        vae.compile(optimizer='adam', loss=vae_loss_function)
        vae.summary()
        print("Loaded model from disk")
    else:
        vae.compile(optimizer='adam', loss=vae_loss_function)
        vae.summary()
        vae.fit(x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks = [print_weights])
        # serialize weights to HDF5
        vae.save_weights("vae_weights.h5")
        print("Saved weights to disk")

    # plot_model(vae, to_file='vae.png', show_shapes=True)
    plot_results(models, test_data, batch_size=batch_size, num_samples=num_samples)
