from keras.layers import Lambda, Input, Dense, Conv1D, UpSampling1D, Flatten
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.engine.topology import Layer

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Constants and params
theta_range = np.deg2rad(120.0)
num_rays = 100
trained = False

H = 9  # no of past observations
F = 1  # no of future predictions

training_data_fraction = 0.8

class RotateLIDAR(Layer):

    def __init__(self, num_rays, 
                 num_past_obs,
                 num_future_pred,
                 **kwargs):
        self.num_rays = num_rays
        self.num_past_obs = num_past_obs
        self.num_future_pred = num_future_pred
        self.output_dim = num_future_pred * num_rays
        super(RotateLIDAR, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(RotateLIDAR, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def prepareDataset(train_file, test_file):
    x_train, x_val, y_train, y_val, x_test, y_test = [], [], [], [], [], []

    if not trained:
        f1 = open(train_file, "r")
        x_raw = []
        y_raw = []
        lines = [line.rstrip('\n') for line in f1]
        for i in range(len(lines) - H - F):
            print "preparing training data point", i
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
                for k in range(num_rays):
                    y.append(float(parts[k + 2]))
            
            x = np.asarray(x)
            u = np.asarray([u])
            x = np.concatenate((x, u.T), axis=1)
            x = x.flatten('F')

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
        print "preparing testing data point", i
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
            for k in range(num_rays):
                y.append(float(parts[k + 2]))
            
        x = np.asarray(x)
        u = np.asarray([u])
        x = np.concatenate((x, u.T), axis=1)
        x = x.flatten('F')

        x_raw.append(x)
        y_raw.append(y)

    x_test = np.asarray(x_raw)
    y_test = np.asarray(y_raw)

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


def plot_results(models,
                 data,
                 batch_size=128,
                 num_samples=1):
    # encoder, decoder = models
    encoder, decoder = models
    x, y = data
    theta_inc = theta_range / float(num_rays)
    for k in range(120, 300, 1):
        plt.figure()
        for i in range(num_samples):
            _, _, z = encoder.predict(np.array([x[k], ]), batch_size=batch_size)
            y_pred = decoder.predict(z, batch_size=batch_size)
            plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y_pred[0], 'b.')
        plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y[k], 'r.')
        plt.ylabel("output")

        '''
        plt.figure()
        x_plot = np.asarray(np.split(x[k][:(H*num_rays)], H))
        for i in range(H):
          plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], x_plot[i][:num_rays], 'b.')
        plt.ylabel("input")
        '''

        plt.show()


x_train, y_train, x_val, y_val, x_test, y_test = prepareDataset(sys.argv[1], sys.argv[2])

x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# network parameters
original_dim = x_test.shape[1]
output_dim = y_test.shape[1]

conv1_filters = 10
dim2 = 202
batch_size = 128
latent_dim = 50
epochs = 5
num_samples = 50
input_shape = (original_dim,1)
output_shape = (output_dim,)

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(original_dim, 1), name='encoder_input')
x1 = Conv1D(conv1_filters, 9, strides=9, activation='relu', input_shape=(None, original_dim))(inputs)
xf = Flatten()(x1)
x2 = Dense(dim2, activation='relu')(xf)
z_mean = Dense(latent_dim, name='z_mean')(x2)
z_log_var = Dense(latent_dim, name='z_log_var')(x2)

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
                validation_data=(x_val, y_val))
        # serialize weights to HDF5
        vae.save_weights("vae_weights.h5")
        print("Saved weights to disk")

    # plot_model(vae, to_file='vae.png', show_shapes=True)
    plot_results(models, test_data, batch_size=batch_size, num_samples=num_samples)
