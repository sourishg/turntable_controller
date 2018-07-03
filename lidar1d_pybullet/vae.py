from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Conv1D, UpSampling1D, Flatten, MaxPooling1D, RepeatVector
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

class VAE:
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
        prev_controls = K.expand_dims(controls[:, :self.H], axis=-1)
        prev_controls = K.tile(prev_controls, [1, 1, 10])
        return Flatten()(prev_controls)

    def _unpack_output(self, y):
        return np.reshape(y, (self.F, self.num_rays), order='F')

    def _build_model(self):
        input_rays_shape = (self.H, self.num_rays)
        input_control_dim = self.H + self.F
        latent_dim = 20

        input_rays = Input(shape=input_rays_shape)
        input_controls = Input(shape=(input_control_dim,))
        inputs = [input_rays, input_controls]

        prev_controls = Lambda(self._get_prev_controls, name='controls')(input_controls)

        x1 = Conv1D(80, 3, activation='relu', padding='same', input_shape=input_rays_shape)(input_rays)
        x2 = Conv1D(40, 3, activation='relu', padding='same')(x1)
        x3 = Conv1D(20, 3, activation='relu', padding='same')(x2)
        x4 = Conv1D(10, 3, activation='relu', padding='same')(x3)
        x5 = Flatten()(x4)

        x6 = Multiply()([x5, prev_controls])
        x7 = Dense(50)(x6)

        '''
        x1 = Dense(input_control_dim, activation='softplus')(input_rays)
        x2 = Multiply()([x1, input_controls])
        x3 = Reshape((input_control_dim, 1), input_shape=(input_control_dim,))(x2)
        x4 = Conv1D(16, 3, activation='relu', padding='same', input_shape=(None, input_control_dim, 1))(x3)
        x5 = MaxPooling1D(2, padding='same')(x4)
        x6 = Conv1D(8, 3, activation='relu', padding='same')(x5)
        x7 = MaxPooling1D(2, padding='same')(x6)
        x8 = Conv1D(4, 3, activation='relu', padding='same')(x7)
        x9 = MaxPooling1D(2, padding='same')(x8)
        x10 = Flatten()(x9)
        x11 = Dense(latent_dim, activation='softplus')(x10)
        '''

        self.z_mean = Dense(latent_dim, name='z_mean')(x7)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x7)
        self.z = Lambda(self._sampling, name='z')([self.z_mean, self.z_log_var])

        self.encoder = Model(inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        self.encoder.summary()

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        y0 = Dense(40, activation='relu')(latent_inputs)
        y1 = Dense(80, activation='relu')(y0)
        outputs = Dense(self.num_rays, activation='relu')(y1)

        '''
        y0 = Dense(latent_dim, activation='softplus')(latent_inputs)
        y1 = Reshape((latent_dim, 1), input_shape=(latent_dim,))(y0)
        y2 = Conv1D(4, 3, activation='relu', padding='same')(y1)
        y3 = UpSampling1D(2)(y2)
        y4 = Conv1D(8, 3, activation='relu', padding='same')(y3)
        y5 = UpSampling1D(2)(y4)
        y6 = Conv1D(16, 3, activation='relu', padding='same')(y5)
        y7 = Flatten()(y6)
        outputs = Dense(output_dim, activation='relu')(y7)
        '''

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
                #print(y_pred.shape)
                y_pred = self._unpack_output(y_pred[0])
                for p in range(self.F):
                    plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y_pred[p]], 'b.')
                #plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y_pred[0], 'b.')
                #print("ypred:", y_pred[0])
            
            for p in range(self.F):
                plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y1[p]], 'r.')
            #plt.ylabel("output")

            '''
            plt.figure()
            x_plot = np.asarray(np.split(x[k][:(H*num_rays)], H))
            for i in range(H):
              plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], x_plot[i][:num_rays], 'b.')
            plt.ylabel("input")
            '''

            plt.show()

    def predict(self, x, u):
        y_pred = []
        for i in range(self.num_samples):
            _, _, z = self.encoder.predict([np.array([x,]),np.array([u,])], batch_size=1)
            y_pred.append(self.decoder.predict(z, batch_size=1)[0])
        y_pred = np.asarray(y_pred)
        y = np.mean(y_pred, axis=0)
        return self._unpack_output(y)