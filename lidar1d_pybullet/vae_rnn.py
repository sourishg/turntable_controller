from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Flatten, LSTM, SimpleRNN, Bidirectional, RepeatVector
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import LambdaCallback
from keras import activations, initializers

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class DeepVAE:
    def __init__(self,
                 num_rays,
                 H, F,
                 var_samples, 
                 epochs=10, batch_size=128):
        self.num_rays = num_rays
        self.H = H
        self.F = F
        self.num_samples = var_samples
        self.epochs = epochs
        self.batch_size = batch_size

    def _vae_loss_function(self, y_true, y_pred):
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        mse_loss = mse(y_true, y_pred)
        mse_loss *= (self.H)
        return K.mean(mse_loss + kl_loss)

    def _sampling(self, args):
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

    def _get_prev_controls(self, controls):
        prev_controls = controls[:, :self.H]
        prev_controls = K.tile(prev_controls, [1, 4])
        return prev_controls

    def _unpack_output(self, y):
        return np.reshape(y, (self.F, self.num_rays), order='F')

    def _build_model(self):
        sess = tf.Session()
        K.set_session(sess)

        input_shape = (self.H, self.num_rays)
        input_control_dim = self.H + self.F
        latent_dim = 20

        input_rays = Input(shape=input_shape)
        input_controls = Input(shape=(input_control_dim,))
        inputs = [input_rays, input_controls]

        x1 = LSTM(80, return_sequences=True, input_shape=input_shape)(input_rays)
        x2 = LSTM(40, return_sequences=False)(x1)

        prev_controls = Lambda(self._get_prev_controls, name='controls')(input_controls)

        x3 = Dense(40)(prev_controls)
        x4 = Multiply()([x2, x3])

        self.z_mean = Dense(latent_dim, name='z_mean')(x4)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x4)
        self.z = Lambda(self._sampling, name='z')([self.z_mean, self.z_log_var])

        self.encoder = Model(inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        self.encoder.summary()

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        y0 = RepeatVector(self.H)(latent_inputs)
        y1 = LSTM(40, return_sequences=True, input_shape=(self.H, latent_dim))(y0)
        y2 = LSTM(80, return_sequences=False)(y1)
        y3 = Dense(self.num_rays, activation='relu')(y2)
        outputs = Dense(self.num_rays, activation='relu')(y3)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        print("Built VAE architecture!")

    def load_weights(self, filename):
        self._build_model()
        self.vae.load_weights(filename)
        self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
        self.vae.summary()
        print("Loaded weights!")

    def fit(self, x_train, x_val, y_train, y_val, u_train, u_val):
        self._build_model()
        self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
        self.vae.summary()
        self.vae.fit([x_train, u_train],
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=([x_val, u_val], y_val))
        # serialize weights to HDF5
        self.vae.save_weights("vae_weights.h5")
        print("Saved weights!")

    def plot_results(self, x_test, u_test, y_test):
        for k in range(y_test.shape[0]):
            fig = plt.figure()
            y1 = self._unpack_output(y_test[k])
            plots = []
            for p in range(self.F):
                ax = fig.add_subplot(2, self.F/2 + 1, p+1)
                plots.append(ax)
            
            for i in range(self.num_samples):
                _, _, z = self.encoder.predict([np.array([x_test[k],]),np.array([u_test[k],])], batch_size=1)
                y_pred = self.decoder.predict(z, batch_size=1)
                y_pred = self._unpack_output(y_pred[0])
                for p in range(self.F):
                    plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y_pred[p]], 'b.')
            
            for p in range(self.F):
                plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y1[p]], 'r.')

            plt.show()

    def predict(self, x, u):
        y_pred = []
        for i in range(self.num_samples):
            _, _, z = self.encoder.predict([np.array([x,]),np.array([u,])], batch_size=1)
            y_pred.append(self.decoder.predict(z, batch_size=1)[0])
        y_pred = np.asarray(y_pred)
        y = np.mean(y_pred, axis=0)
        return self._unpack_output(y)