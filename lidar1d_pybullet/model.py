from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Conv1D, UpSampling1D, Flatten, MaxPooling1D, RepeatVector, LSTM, Add, TimeDistributed, Concatenate
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import LambdaCallback
from keras import activations

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class TRFModel:
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

        self.input_rays_shape = (self.H, self.num_rays)
        self.input_control_shape = (self.H + self.F, )
        self.latent_dim = 10

    def _vae_loss_function(self, y_true, y_pred):
        kl_loss = -0.5 * (1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var))
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss = K.mean(kl_loss, axis=-1)
        mse_loss = mse(y_true, y_pred)
        mse_loss *= (self.H)
        return mse_loss + kl_loss
        # return mse_loss

    def _sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
          args (tensor): mean and log of variance of Q(z|X)
        # Returns:
          z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        steps = K.int_shape(z_mean)[1]
        dim = K.int_shape(z_mean)[2]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, steps, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _get_prev_controls(self, controls):
        prev_controls = K.expand_dims(controls[:, :self.H], axis=-1)
        return K.tile(prev_controls, [1, 1, 10])

    def _unpack_output(self, y):
        return np.reshape(y, (self.F, self.num_rays), order='F')

    def _build_latent_model(self, input_rays):
        x1 = LSTM(80, return_sequences=True, input_shape=self.input_rays_shape)(input_rays)
        x2 = LSTM(40, return_sequences=True)(x1)
        x3 = LSTM(20, return_sequences=True)(x2)

        self.z_mean = TimeDistributed(Dense(self.latent_dim, name='z_mean', input_shape=(self.H, 20)))(x3)
        self.z_log_var = TimeDistributed(Dense(self.latent_dim, name='z_log_var', input_shape=(self.H, 20)))(x3)
        self.z = Lambda(self._sampling, name='z')([self.z_mean, self.z_log_var])

        self.encoder = Model(input_rays, [self.z_mean, self.z_log_var, self.z], name='latent_model')
        self.encoder.summary()

    def _build_initial_model(self, input_rays, input_controls):
        z = self.encoder(input_rays)[2]
        prev_controls = Lambda(self._get_prev_controls, name='controls')(input_controls)
        zc = Concatenate()([z, prev_controls])

        x1 = LSTM(100, return_sequences=True, input_shape=self.input_rays_shape)(input_rays)
        x2 = LSTM(100, return_sequences=True)(x1)
        x3 = LSTM(80, return_sequences=True)(x2)
        x4 = Concatenate()([x3, zc])
        x5 = LSTM(100, return_sequences=False)(x4)
        # x6 = Dense(100, activation='sigmoid')(x5)
        
        self.vae = Model([input_rays, input_controls], x5, name='vae')
        self.vae.summary()

    def _build_model(self):
        input_rays = Input(shape=self.input_rays_shape)
        input_controls = Input(shape=self.input_control_shape)

        self._build_latent_model(input_rays)
        self._build_initial_model(input_rays, input_controls)

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
        for k in range(500, y_test.shape[0], 1):
            fig = plt.figure()
            y1 = self._unpack_output(y_test[k])
            plots = []
            for p in range(self.F):
                ax = fig.add_subplot(2, self.F/2 + 1, p+1)
                plots.append(ax)
            
            for i in range(self.num_samples):
                #_, _, z = self.encoder.predict([np.array([x_test[k],]),np.array([u_test[k],])], batch_size=1)
                #y_pred = self.decoder.predict(z, batch_size=1)
                #print(y_pred.shape)
                y_pred = self.vae.predict([np.array([x_test[k],]),np.array([u_test[k],])], batch_size=1)
                y_pred = self._unpack_output(y_pred[0])
                for p in range(self.F):
                    plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y_pred[p]], 'b.')
                #plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y_pred[0], 'b.')
                #print("ypred:", y_pred[0])
            
            for p in range(self.F):
                plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y1[p]], 'r.')

            plt.show()

    def predict(self, x, u):
        '''
        y_pred = []
        for i in range(self.num_samples):
            _, _, z = self.encoder.predict([np.array([x,]),np.array([u,])], batch_size=1)
            y_pred.append(self.decoder.predict(z, batch_size=1)[0])
        y_pred = np.asarray(y_pred)
        y = np.mean(y_pred, axis=0)
        return self._unpack_output(y)
        '''