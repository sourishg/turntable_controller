from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Flatten, Dot, Add, TimeDistributed, Concatenate, CuDNNLSTM
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.constraints import non_neg

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

        self.control_dim = 1
        self.latent_dim = 20

        self.encoder = None
        self.transition_model = None
        self.cost_model = None
        self.vae = None

        self.kl_loss = 0
        self.trans_loss = 0

        self.training_phase = 0

    def _get_task_relevant_cost(self, x):
        lo = int(FLAGS.num_rays) / 2 - int(FLAGS.tr_half_width)
        hi = int(FLAGS.num_rays) / 2 + int(FLAGS.tr_half_width)
        return K.max(x[:, lo:hi], axis=-1, keepdims=True)

    def _compute_kl_loss(self, z_mean, z_log_var):
        loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        loss = K.sum(loss, axis=-1)
        return (-0.5) * loss

    def _vae_loss_function(self, y_true, y_pred):
        self.kl_loss = self.kl_loss * FLAGS.latent_multiplier

        self.trans_loss = self.trans_loss * 1e-03

        # mse_loss = mse(y_true, y_pred)
        delta_y = y_pred - y_true
        error_y = K.switch(K.greater_equal(delta_y, 0), lambda: 0.6 * delta_y, lambda: -delta_y)
        mse_loss = K.mean(K.square(error_y), axis=-1)

        if self.training_phase == 0:
            return mse_loss + self.kl_loss
        elif self.training_phase == 1:
            return mse_loss + self.kl_loss + self.trans_loss
        else:
            return mse_loss + self.kl_loss + self.trans_loss

    def _sampling_prior(self, args):
        rays = args
        batch = K.shape(rays)[0]
        dim = self.latent_dim
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

    def _build_encoder_model(self):
        input_ray = Input(shape=(self.num_rays, ), name='input_ray')

        enc1 = Dense(80, activation='relu')(input_ray)
        enc2 = Dense(80, activation='relu')(enc1)
        enc3 = Dense(40, activation='relu')(enc2)
        enc4 = Dense(40, activation='relu')(enc3)

        z_mean = Dense(self.latent_dim, activation='relu')(enc4)
        z_std = Dense(self.latent_dim, activation='relu')(enc4)
        z_log_var = Lambda(lambda x: x + FLAGS.latent_std_min)(z_std)

        z = Lambda(self._sampling)([z_mean, z_log_var])

        return Model(input_ray, [z_mean, z_log_var, z], name='encoder_model')

    def _build_transition_model(self):
        prev_latent_state = Input(shape=(self.latent_dim, ), name='prev_latent_state')
        control = Input(shape=(self.control_dim, ), name='input_control')

        z = Dense(self.latent_dim, use_bias=False)(prev_latent_state)
        u = Dense(self.latent_dim, use_bias=False)(control)
        outputs = Add()([z, u])

        return Model([prev_latent_state, control], outputs, name='transition_model')

    def _build_cost_model(self):
        latent_state = Input(shape=(self.latent_dim,), name='latent_state')
        control = Input(shape=(self.control_dim,), name='control')

        u = Dense(self.latent_dim, use_bias=False)(latent_state)
        v = Dense(self.control_dim, use_bias=False)(control)
        r = Lambda(lambda x: K.batch_dot(x, x, axes=1))(u)
        s = Lambda(lambda x: K.batch_dot(x, x, axes=1))(v)
        cost = Add()([r, s])

        """
        p = Dense(self.latent_dim, kernel_constraint=non_neg(), use_bias=False)(latent_state)
        q = Dense(1, kernel_constraint=non_neg(), use_bias=False)(control)
        r = Dot(axes=1)([latent_state, p])
        s = Dot(axes=1)([control, q])
        cost = Add()([r, s])
        """

        return Model([latent_state, control], cost, name='cost_model')

    def _build_vae_model(self):
        input_rays = Input(shape=(self.H + self.F, self.num_rays), name='input_rays')
        input_controls = Input(shape=(self.H + self.F, ), name='input_controls')

        self.encoder = self._build_encoder_model()
        self.transition_model = self._build_transition_model()
        self.cost_model = self._build_cost_model()
        # self.cost_model.layers[-1].trainable_weights.extend([W, b])

        outputs = []

        # ray_init = Lambda(lambda x: x[:, 0, :])(input_rays)
        # z_mean, z_log_var, z_init = self.encoder(ray_init)

        for i in range(1, self.H + self.F, 1):
            ray = Lambda(lambda x: x[:, i, :])(input_rays)
            ray_prev = Lambda(lambda x: x[:, i - 1, :])(input_rays)
            u_prev = Lambda(lambda x: K.expand_dims(x[:, i - 1], axis=-1))(input_controls)
            u = Lambda(lambda x: K.expand_dims(x[:, i], axis=-1))(input_controls)

            if self.training_phase == 0:
                z_mean, z_log_var, z = self.encoder(ray)
                self.kl_loss = self.kl_loss + self._compute_kl_loss(z_mean, z_log_var)
                y = self.cost_model([z, u_prev])

            if self.training_phase == 1:
                _, _, prev_z = self.encoder(ray_prev)
                z_mean, z_log_var, true_z = self.encoder(ray)
                self.kl_loss = self.kl_loss + self._compute_kl_loss(z_mean, z_log_var)

                z = self.transition_model([prev_z, u_prev])
                self.trans_loss = self.trans_loss + K.square(true_z - z)
                y = self.cost_model([z, u_prev])

            if self.training_phase == 2:
                _, _, prev_z = self.encoder(ray_prev)
                z_mean, z_log_var, true_z = self.encoder(ray)
                self.kl_loss = self.kl_loss + self._compute_kl_loss(z_mean, z_log_var)

                if i < self.H:
                    z = self.transition_model([prev_z, u_prev])
                    self.trans_loss = self.trans_loss + K.square(true_z - z)
                    y = self.cost_model([z, u_prev])
                else:
                    z = self.transition_model([z, u_prev])
                    self.trans_loss = self.trans_loss + K.square(true_z - z)
                    y = self.cost_model([z, u_prev])

            outputs.append(y)

        if self.training_phase > 0:
            self.trans_loss = K.mean(self.trans_loss, axis=-1)

        outputs_flat = outputs[0]
        for i in range(1, self.F + self.H - 1, 1):
            outputs_flat = Concatenate()([outputs_flat, outputs[i]])

        return Model([input_rays, input_controls], outputs_flat, name='vae_model')

    def load_weights(self, filename):
        self.training_phase = 2
        self.vae = self._build_vae_model()
        self.vae.load_weights(filename, by_name=True)
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

        if self.training_phase == 1:
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

            """
            self.transition_model.compile(optimizer='adam', loss='mean_squared_error')

            for i in range(1, self.H + self.F, 1):
                ray = x_train[:, i, :]
                ray_prev = x_train[:, i - 1, :]
                u_prev = u_train[:, i - 1]

                prev_belief = self.encoder.predict(ray_prev)
                true_belief = self.encoder.predict(ray)

                print(prev_belief.shape, u_prev.shape)

                self.transition_model.fit([prev_belief, u_prev],
                                          true_belief,
                                          batch_size=1000,
                                          epochs=10)
            """

        if self.training_phase == 2:
            print("Training phase:", self.training_phase)
            self.vae = self._build_vae_model()
            if self.task_relevant:
                self.vae.load_weights("vae_weights_tr_p1.h5", by_name=True)
            else:
                self.vae.load_weights("vae_weights_p1.h5", by_name=True)
            self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
            self.vae.summary()
            self.vae.fit([x_train, u_train],
                         y_train,
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_data=([x_val, u_val], y_val))
            # serialize weights to HDF5
            if self.task_relevant:
                self.vae.save_weights("vae_weights_tr_p2.h5")
            else:
                self.vae.save_weights("vae_weights_p2.h5")
            print("Saved weights phase", self.training_phase)

    def get_encoder_model(self):
        return self.encoder

    def get_transition_model(self):
        return self.transition_model

    def get_cost_model(self):
        return self.cost_model

    def get_vae_model(self):
        return self.vae
