from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Flatten, Dropout, Add, TimeDistributed, Concatenate, CuDNNLSTM
from keras.models import Model
from keras.losses import mse
from keras import backend as K

import params
import numpy as np
import tensorflow as tf

FLAGS = params.FLAGS

class TRFModel:
    def __init__(self,
                 num_rays,
                 H, F,
                 var_samples, 
                 epochs=10, batch_size=128,
                 task_relevant=False):
        self.num_rays = num_rays
        self.H = H
        self.F = F
        self.num_samples = var_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.task_relevant = task_relevant

        if self.task_relevant:
            self.output_dim = 1
        else:
            self.output_dim = self.num_rays

        self.latent_dim = 10

        self.encoder = None
        self.decoder = None
        self.transition_model = None
        self.vae = None

        self.kl_loss = 0
        self.trans_loss = 0

        self.training_phase = 1

    def _get_task_relevant_feature(self, y):
        lo = int(FLAGS.num_rays) / 2 - int(FLAGS.tr_half_width)
        hi = int(FLAGS.num_rays) / 2 + int(FLAGS.tr_half_width)
        return K.max(y[:, lo:hi], axis=-1, keepdims=True)

    def _compute_kl_loss(self, z_mean, z_log_var):
        loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        loss = K.sum(loss, axis=-1)
        return (-0.5) * loss

    def _vae_loss_function(self, y_true, y_pred):
        self.kl_loss = self.kl_loss * FLAGS.latent_multiplier

        # self.trans_loss = self.trans_loss * 0.001

        # mse_loss = mse(y_true, y_pred)
        delta_y = y_pred - y_true
        error_y = K.switch(K.greater_equal(delta_y, 0), lambda: 0.8 * delta_y, lambda: -delta_y)
        mse_loss = K.mean(K.square(error_y), axis=-1)

        if self.training_phase == 0:
            return mse_loss + self.kl_loss
        else:
            return mse_loss + self.trans_loss

    def _sampling_prior(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        return K.random_normal(shape=(batch, dim))

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

    def _build_encoder_model(self):
        input_ray = Input(shape=(self.num_rays,), name='input_ray')

        enc1 = Dense(80, input_shape=(None, self.num_rays))(input_ray)
        enc2 = Dense(60)(enc1)
        enc3 = Dense(40)(enc2)
        enc4 = Dense(20)(enc3)
        z_mean = Dense(self.latent_dim)(enc4)
        z_std = Dense(self.latent_dim)(enc4)

        z_log_var = Lambda(lambda x: x + FLAGS.latent_std_min)(z_std)
        z = Lambda(self._sampling)([z_mean, z_log_var])

        return Model(input_ray, [z_mean, z_log_var, z], name='encoder_model')

    def _build_transition_model(self):
        z = Input(shape=(self.latent_dim,), name='z_input')
        control = Input(shape=(1,), name='u_input')
        u = Lambda(lambda x: K.tile(x, [1, self.latent_dim]))(control)

        zu = Concatenate()([z, u])
        d1 = Dense(50)(zu)
        d2 = Dense(50)(d1)
        d3 = Dense(50)(d2)
        outputs = Dense(self.latent_dim)(d3)

        return Model([z, control], outputs, name='transition_model')

    def _build_decoder_model(self):
        z = Input(shape=(self.latent_dim,), name='z_input')

        dec0 = Dense(20, activation='tanh')(z)
        dec1 = Dense(40, activation='tanh')(dec0)
        dec2 = Dense(60, activation='tanh')(dec1)
        dec3 = Dense(80, activation='tanh')(dec2)
        dec4 = Dense(self.num_rays, activation='relu')(dec3)
        if self.task_relevant:
            outputs = Lambda(self._get_task_relevant_feature)(dec4)
        else:
            outputs = Dense(self.output_dim, activation='relu')(dec4)

        return Model(z, outputs, name='decoder_model')

    def _build_vae_model(self):
        input_rays = Input(shape=(self.H + self.F, self.num_rays), name='input_rays')
        input_controls = Input(shape=(self.H + self.F, ), name='input_controls')

        self.encoder = self._build_encoder_model()
        self.decoder = self._build_decoder_model()
        self.transition_model = self._build_transition_model()

        if self.training_phase == 0:
            for layer in self.transition_model.layers:
                layer.trainable = False

        if self.training_phase > 0:
            for layer in self.encoder.layers:
                layer.trainable = False
            for layer in self.decoder.layers:
                layer.trainable = False

        outputs = []

        ray_init = Lambda(lambda x: x[:, 0, :])(input_rays)
        zp_mean, zp_std, zp = self.encoder(ray_init)

        for i in range(1, self.H + self.F, 1):
            ray = Lambda(lambda x: x[:, i, :])(input_rays)
            z_mean, z_log_var, z = self.encoder(ray)

            self.kl_loss = self.kl_loss + self._compute_kl_loss(z_mean, z_log_var)

            if self.training_phase == 1:
                ray_prev = Lambda(lambda x: x[:, i - 1, :])(input_rays)
                u_prev = Lambda(lambda x: K.expand_dims(x[:, i - 1], axis=-1))(input_controls)
                z_mean_prev, z_log_var_prev, z_prev = self.encoder(ray_prev)
                zp = self.transition_model([z_prev, u_prev])
                # zp = Lambda(self._sampling)([zp_mean, zp_std])
                y = self.decoder(zp)

                self.trans_loss = self.trans_loss + K.square(z - zp)

            if self.training_phase == 0:
                y = self.decoder(z)

            outputs.append(y)

        if self.training_phase == 1:
            self.trans_loss = K.mean(self.trans_loss, axis=-1)

        outputs_flat = outputs[0]
        for i in range(1, self.F + self.H - 1, 1):
            outputs_flat = Concatenate()([outputs_flat, outputs[i]])

        return Model([input_rays, input_controls], outputs_flat, name='vae_model')

    def load_weights(self, filename):
        self.training_phase = 0
        self.vae = self._build_vae_model()
        self.vae.load_weights(filename)
        self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
        self.vae.summary()
        print("Loaded weights!")

    def train_model(self, x_train, x_val, y_train, y_val, u_train, u_val):
        if self.training_phase == 0:
            print("Training phase:", self.training_phase)
            self.vae = self._build_vae_model()
            self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
            self.vae.summary()
            self.vae.fit([x_train, u_train],
                         y_train,
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_data=([x_val, u_val], y_val))
            # serialize weights to HDF5
            if self.task_relevant:
                self.vae.save_weights("vae_weights_tr_p0.h5")
            else:
                self.vae.save_weights("vae_weights_p0.h5")
            print("Saved weights phase", self.training_phase)

        self.training_phase = 1
        print("Training phase:", self.training_phase)
        self.vae = self._build_vae_model()
        if self.task_relevant:
            self.vae.load_weights("vae_weights_tr_p0.h5", by_name=True)
        else:
            self.vae.load_weights("vae_weights_p0.h5", by_name=True)
        self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
        self.vae.summary()
        self.vae.fit([x_train, u_train],
                     y_train,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     validation_data=([x_val, u_val], y_val))
        # serialize weights to HDF5
        if self.task_relevant:
            self.vae.save_weights("vae_weights_tr_p1.h5")
        else:
            self.vae.save_weights("vae_weights_p1.h5")
        print("Saved weights phase", self.training_phase)

    def get_encoder_model(self):
        return self.encoder

    def get_decoder_model(self):
        return self.decoder

    def get_transition_model(self):
        return self.transition_model

    def get_vae_model(self):
        return self.vae
