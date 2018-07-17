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

        if FLAGS.task_relevant:
            self.output_dim = 1
            self.latent_dim = 1
        else:
            self.output_dim = self.num_rays
            self.latent_dim = 10

    def _compute_kl_loss(self, z_mean, z_log_var):
        loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        loss = K.sum(loss, axis=-1)
        return (-0.5) * loss

    def _vae_loss_function(self, y_true, y_pred):
        kl_loss = self._compute_kl_loss(self.z_mean, self.z_log_var)
        kl_loss = 1e-04 * K.mean(kl_loss, axis=-1)
        
        mse_loss = mse(y_true, y_pred)
        return mse_loss + kl_loss
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
        return np.reshape(y, (self.H + self.F - 1, self.output_dim))
        # return np.reshape(y, (self.F, self.output_dim))

    def _build_latent_model(self, input_rays, input_controls):
        enc1 = CuDNNLSTM(80, return_sequences=True, input_shape=(None, None, self.num_rays))(input_rays)
        enc2 = CuDNNLSTM(40, return_sequences=True)(enc1)
        enc3 = CuDNNLSTM(20, return_sequences=True)(enc2)

        self.z_mean = TimeDistributed(Dense(self.latent_dim, name='z_mean'), input_shape=(K.int_shape(enc3)[1], K.int_shape(enc3)[2]))(enc3)
        z_std = TimeDistributed(Dense(self.latent_dim, name='z_log_var'), input_shape=(K.int_shape(enc3)[1], K.int_shape(enc3)[2]))(enc3)

        self.z_log_var = Lambda(lambda x: x - 5.0)(z_std)

        return Model([input_rays, input_controls], [self.z_mean, self.z_log_var], name='latent_model')

    def _build_generative_model(self):
        prev_y = Input(shape=(self.output_dim,))
        control = Input(shape=(1,))
        z = Input(shape=(self.latent_dim,))
        u = Lambda(lambda x: K.tile(x, [1, self.latent_dim]))(control)
        zu = Concatenate()([z, u])
        dec1 = Dense(self.output_dim, activation='tanh')(zu)
        dec2 = Reshape((self.output_dim, 1), input_shape=(self.output_dim,))(dec1)

        y = Lambda(lambda x: K.expand_dims(x, axis=-1))(prev_y)
        dec3 = CuDNNLSTM(1, return_sequences=True, input_shape=(self.output_dim, 1))(y)
        dec4 = CuDNNLSTM(1, return_sequences=True)(dec3)
        dec5 = CuDNNLSTM(1, return_sequences=True)(dec4)
        dec6 = Multiply()([dec2, dec5])
        dec8 = Reshape((self.output_dim,), input_shape=(self.output_dim, 1))(dec6)
        outputs = Dense(self.output_dim, activation='relu')(dec8)

        return Model([prev_y, control, z], outputs)

    def _build_model(self):
        input_rays = Input(shape=(self.H + self.F, self.num_rays))
        input_controls = Input(shape=(self.H + self.F, ))

        self.encoder = self._build_latent_model(input_rays, input_controls)
        self.encoder.summary()
        
        self.gen_model = self._build_generative_model()
        z_mean, z_std = self.encoder([input_rays, input_controls])
        outputs = []
        prev_y, pred_y = None, None
        for i in range(self.H + self.F - 1):
            if i >= self.H:
                prev_y = pred_y
            else:
                prev_y = Lambda(lambda x: x[:, i, :])(input_rays)
            z_mean_t = Lambda(lambda x: x[:, i+1, :])(z_mean)
            z_std_t = Lambda(lambda x: x[:, i+1, :])(z_std)
            z = Lambda(self._sampling)([z_mean_t, z_std_t])
            u = Lambda(lambda x : K.expand_dims(x[:, i], axis=-1))(input_controls)
            pred_y = self.gen_model([prev_y, u, z])
            outputs.append(pred_y)

        outputs_flat = outputs[0]
        for i in range(1, self.F + self.H - 1, 1):
            outputs_flat = Concatenate()([outputs_flat, outputs[i]])

        self.vae = Model([input_rays, input_controls], outputs_flat, name='vae')
        self.vae.summary()

        print("Built VAE architecture!")

    def _build_model_testing(self):
        input_rays = Input(shape=(self.H, self.num_rays))
        input_controls = Input(shape=(self.H + self.F, ))

        input_controls_enc = Lambda(lambda x : x[:, :self.H])(input_controls)

        self.encoder = self._build_latent_model(input_rays, input_controls)
        self.encoder.summary()
        
        self.gen_model = self._build_generative_model()
        z_mean, z_std = self.encoder([input_rays, input_controls])
        outputs = []
        prev_y = Lambda(lambda x : x[:, self.H-1, :])(input_rays)
        for i in range(self.F):
            z_mean_t = Lambda(lambda x : x[:, self.H-1, :])(z_mean)
            z_std_t = Lambda(lambda x : x[:, self.H-1, :])(z_std)
            u = Lambda(lambda x : K.expand_dims(x[:, self.H-1+i], axis=-1))(input_controls)
            pred_y = self.gen_model([prev_y, u, z_mean_t, z_std_t])
            prev_y = pred_y
            outputs.append(pred_y)

        outputs_flat = outputs[0]
        for i in range(1, self.F, 1):
            outputs_flat = Concatenate()([outputs_flat, outputs[i]])

        self.vae = Model([input_rays, input_controls], outputs_flat, name='vae')
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
            fig = plt.figure(figsize=(15, 8))
            y1 = self._unpack_output(y_test[k])
            plots = []
            num_plots = self.H + self.F - 1
            # num_plots = self.F
            for p in range(num_plots):
                ax = fig.add_subplot(3, num_plots/3 + 1, p+1)
                ax.set_ylim([0, 1.0])
                ax.set_title("Timestep " + str(p+1))
                plots.append(ax)
            
            for i in range(self.num_samples):
                #_, _, z = self.encoder.predict(np.array([x_test[k]]), batch_size=1)
                #y_pred = self.decoder.predict([np.array([x_test[k]]),np.array([u_test[k]]),z], batch_size=1)
                #print(y_pred.shape)
                y_pred = self.vae.predict([np.array([x_test[k],]),np.array([u_test[k],])], batch_size=1)
                y_pred = self._unpack_output(y_pred[0])
                for p in range(num_plots):
                    plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y_pred[p]], 'b.')
                #plt.plot([j * np.rad2deg(theta_inc) for j in range(num_rays)], y_pred[0], 'b.')
                #print("ypred:", y_pred[0])
            
            for p in range(num_plots):
                plots[p].plot([j for j in range(self.num_rays)], [float(u) for u in y1[p]], 'r.')

            plt.show()

    def predict(self, x, u):
        y_pred = []
        for i in range(self.num_samples):
            y_pred.append(self.vae.predict([np.array([x,]), np.array([u,])], batch_size=1)[0])
        y_pred = np.asarray(y_pred)
        y = np.amax(y_pred, axis=0)
        return self._unpack_output(y)