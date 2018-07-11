from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Conv1D, UpSampling1D, Flatten, MaxPooling1D, RepeatVector, LSTM, Add, TimeDistributed, Concatenate, CuDNNLSTM, Dropout
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import LambdaCallback
from keras import activations

import tensorflow as tf
from tensorflow.python.platform import flags

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

FLAGS = flags.FLAGS

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
        if FLAGS.task_relevant:
            self.output_dim = 1
            self.latent_dim = 1
        else:
            self.output_dim = self.num_rays
            self.latent_dim = 2

    def _compute_kl_loss(self, z_mean, z_log_var):
        loss = -0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        loss = K.sum(loss, axis=-1)
        return loss

    def _vae_loss_function(self, y_true, y_pred):
        kl_loss = self._compute_kl_loss(self.z_mean, self.z_log_var)
        # kl_loss = K.mean(kl_loss, axis=-1)
        
        mse_loss = mse(y_true, y_pred)
        return K.mean(mse_loss + kl_loss)
        # return mse_loss
        # return kl_loss

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
        prev_controls = K.tile(controls[:, :self.H], [1, self.latent_dim])
        return prev_controls

    def _unpack_output(self, y):
        return np.reshape(y, (self.F, self.output_dim))

    def _build_latent_model(self, input_rays, input_controls):
        enc2 = CuDNNLSTM(80, return_sequences=True, input_shape=self.input_rays_shape)(input_rays)
        enc3 = CuDNNLSTM(40, return_sequences=True)(enc2)
        enc4 = CuDNNLSTM(20, return_sequences=True)(enc3)
        enc5 = CuDNNLSTM(self.latent_dim, return_sequences=True)(enc4)

        prev_controls = Lambda(self._get_prev_controls)(input_controls)
        enc6 = Multiply()([Flatten()(enc5), prev_controls])

        self.z_mean = Dense(self.H * self.latent_dim, name='z_mean')(enc6)
        self.z_log_var = Dense(self.H * self.latent_dim, name='z_log_var')(enc6)

        return Model([input_rays, input_controls], [self.z_mean, self.z_log_var], name='latent_model')

    def _state_transition(self, y, z, u):
        '''
        u1 = Lambda(lambda x : K.tile(x, [1, self.H * self.latent_dim]))(u)
        z1 = Multiply()([z, u1])
        z2 = Dense(self.num_rays, activation='tanh')(z1)
        z3 = Add()([z2, y])
        z_out = Dense(self.H * self.latent_dim)(z3)
        '''
        y1 = Dense(self.H * self.latent_dim)(y)
        u1 = Dense(self.H * self.latent_dim)(u)
        z1 = Dense(self.H * self.latent_dim)(z)
        return Add()([z1, u1, y1])

    def _forward_model(self, prev_y, z_mean, z_std, u):
        outputs = None
        if FLAGS.task_relevant:
            z = Lambda(self._sampling)([z_mean, z_std])
            zu = Concatenate()([z, u])
            dec1 = Dense(self.num_rays, activation='tanh')(zu)
            dec2 = Dense(self.num_rays, activation='relu')(prev_y)
            dec3 = Add()([dec1, dec2])
            dec4 = Dense(50, activation='tanh')(dec3)
            outputs = Dense(self.output_dim, activation='relu')(dec4)
        else:
            z = Lambda(self._sampling)([z_mean, z_std])
            zu = Concatenate()([z, u])
            dec1 = Dense(self.output_dim, activation='tanh')(zu)
            y1 = Lambda(lambda x : K.expand_dims(x, axis=-1))(prev_y)
            dec2 = Reshape((self.output_dim, 1), input_shape=(self.output_dim,))(dec1)
            # u = Lambda(lambda x : K.tile(x, [1, self.H * self.latent_dim]))(u)
            dec3 = Add()([dec2, y1])
            dec4 = CuDNNLSTM(1, return_sequences=True, input_shape=(self.output_dim, 1))(dec3)
            dec5 = CuDNNLSTM(1, return_sequences=True)(dec4)
            dec6 = CuDNNLSTM(1, return_sequences=True)(dec5)
            outputs = Reshape((self.output_dim,), input_shape=(self.output_dim, 1))(dec6)
        return outputs

    def _build_transition_model(self, input_rays, input_controls):
        latent_mean = Input(shape=(self.H * self.latent_dim,))
        latent_std = Input(shape=(self.H * self.latent_dim,))
        z_mean = latent_mean
        z_std = latent_std
        prev_y = Lambda(lambda x : x[:, -1, :])(input_rays)
        outputs = []

        # if FLAGS.task_relevant:
        #    prev_y = Dense(self.output_dim)(prev_y)

        for i in range(self.F):
            u = Lambda(lambda x : K.expand_dims(x[:, self.H + i], axis=-1))(input_controls)
            y_pred = self._forward_model(prev_y, z_mean, z_std, u)
            outputs.append(y_pred)
            if i != self.F - 1:
                z_mean = self._state_transition(y_pred, z_mean, u)
                z_std = self._state_transition(y_pred, z_std, u)
                prev_y = y_pred

        outputs_flat = outputs[0]
        for i in range(1, self.F, 1):
            outputs_flat = Concatenate()([outputs_flat, outputs[i]])

        return Model([input_rays, input_controls, latent_mean, latent_std], outputs_flat, name='decoder')

    def _build_model(self):
        input_rays = Input(shape=self.input_rays_shape)
        input_controls = Input(shape=self.input_control_shape)

        self.encoder = self._build_latent_model(input_rays, input_controls)
        self.encoder.summary()

        self.decoder = self._build_transition_model(input_rays, input_controls)
        self.decoder.summary()

        latent_mean, latent_std = self.encoder([input_rays, input_controls])
        outputs = self.decoder([input_rays, input_controls, latent_mean, latent_std])
        
        self.vae = Model([input_rays, input_controls], outputs, name='vae')
        self.vae.summary()

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
        if FLAGS.task_relevant:
            self.vae.save_weights("vae_weights_tr.h5")
        else:
            self.vae.save_weights("vae_weights.h5")
        print("Saved weights!")

    def plot_tr_results(self, x_test, u_test, y_test):
        for k in range(0, y_test.shape[0], 1):
            fig = plt.figure()
            plt.ylim((-1.0, 1.0))
            y_true = y_test[k]
            for i in range(self.num_samples):
                #_, _, z = self.encoder.predict(np.array([x_test[k]]), batch_size=1)
                #y_pred = self.decoder.predict([np.array([x_test[k]]),np.array([u_test[k]]),z], batch_size=1)
                #print(y_pred.shape)
                y_pred = self.vae.predict([np.array([x_test[k],]),np.array([u_test[k],])], batch_size=1)[0]
                plt.plot([j for j in range(self.F)], [float(u) for u in y_pred], 'b.')
                #plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y_pred[0], 'b.')
                #print("ypred:", y_pred[0])
            
            plt.plot([j for j in range(self.F)], [float(u) for u in y_true], 'r.')

            plt.show()

    def plot_results(self, x_test, u_test, y_test):
        for k in range(0, y_test.shape[0], 1):
            fig = plt.figure()
            plt.ylim((0.0, 1.0))
            y1 = self._unpack_output(y_test[k])
            plots = []
            for p in range(self.F):
                ax = fig.add_subplot(2, self.F/2 + 1, p+1)
                plots.append(ax)
            
            for i in range(self.num_samples):
                #_, _, z = self.encoder.predict(np.array([x_test[k]]), batch_size=1)
                #y_pred = self.decoder.predict([np.array([x_test[k]]),np.array([u_test[k]]),z], batch_size=1)
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
        y_pred = []
        for i in range(self.num_samples):
            y_pred.append(self.vae.predict([np.array([x,]), np.array([u,])], batch_size=1)[0])
        y_pred = np.asarray(y_pred)
        y = np.mean(y_pred, axis=0)
        return self._unpack_output(y)