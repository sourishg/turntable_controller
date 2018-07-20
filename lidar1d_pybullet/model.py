from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Multiply, Flatten, LSTM, Add, TimeDistributed, Concatenate, CuDNNLSTM
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
        self.gen_model = None
        self.vae = None

        self.training_phase = 0

        self.sess = tf.Session()
        K.set_session(self.sess)

    def _get_task_relevant_feature(self, y):
        lo = int(FLAGS.num_rays) / 2 - int(FLAGS.tr_half_width)
        hi = int(FLAGS.num_rays) / 2 + int(FLAGS.tr_half_width)
        return K.max(y[:, lo:hi], axis=-1, keepdims=True)

    def _compute_kl_loss(self, z_mean, z_log_var):
        loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        loss = K.sum(loss, axis=-1)
        return (-0.5) * loss

    def _vae_loss_function(self, y_true, y_pred):
        kl_loss = 0
        if self.training_phase == 2:
            kl_loss = self._compute_kl_loss(self.z_mean, self.z_log_var)
            kl_loss = FLAGS.latent_multiplier * kl_loss
        
        # mse_loss = mse(y_true, y_pred)
        delta_y = y_pred - y_true
        error_y = K.switch(K.greater_equal(delta_y, 0), lambda: 0.6 * delta_y, lambda: -delta_y)
        mse_loss = K.mean(K.square(error_y), axis=-1)

        return mse_loss + kl_loss
        # return mse_loss
        # return kl_loss

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

    def _build_latent_model(self, input_rays):
        enc1 = TimeDistributed(Dense(80), input_shape=(None, None, self.num_rays))(input_rays)
        enc2 = TimeDistributed(Dense(40))(enc1)
        enc3 = TimeDistributed(Dense(20))(enc2)
        enc4 = Flatten()(enc3)

        z_mean = Dense(self.latent_dim)(enc4)
        z_std = Dense(self.latent_dim)(enc4)

        z_log_var = Lambda(lambda x: x + FLAGS.latent_std_min)(z_std)

        return Model(input_rays, [z_mean, z_log_var], name='latent_model')

    def _build_generative_model(self):
        prev_y = Input(shape=(self.output_dim,), name='prev_y')
        control = Input(shape=(1,), name='control_input')
        z = Input(shape=(self.latent_dim,), name='latent_input')

        u = Lambda(lambda x: K.tile(x, [1, self.latent_dim]))(control)
        zu = Concatenate()([z, u])

        if self.task_relevant:
            dec3 = Dense(self.latent_dim)(zu)
            dec4 = Dense(self.output_dim)(dec3)
            dec6 = Concatenate()([prev_y, dec4])
            dec7 = Dense(self.output_dim, activation='tanh')(dec6)
            outputs = Dense(self.output_dim, activation='relu')(dec7)
        else:
            dec1 = Dense(self.output_dim)(prev_y)
            dec2 = Dense(self.output_dim)(dec1)
            dec3 = Dense(self.output_dim)(dec2)
            dec4 = Concatenate()([dec3, zu])
            dec5 = Dense(self.output_dim)(dec4)
            dec6 = Dense(self.output_dim, activation='tanh')(dec5)
            outputs = Dense(self.output_dim, activation='relu')(dec6)

        return Model([prev_y, control, z], outputs, name='generative_model')

    def _build_transition_encoder(self, input_prev_rays):
        enc1 = TimeDistributed(Dense(80), input_shape=(None, None, self.num_rays))(input_prev_rays)
        enc2 = TimeDistributed(Dense(40))(enc1)
        enc3 = TimeDistributed(Dense(20))(enc2)
        enc4 = Flatten()(enc3)

        z = Dense(self.num_rays)(enc4)

        return Model(input_prev_rays, z, name='transition_encoder')

    def _build_vae_model(self):
        input_rays = Input(shape=(self.H + self.F, self.num_rays), name='input_rays')
        input_controls = Input(shape=(self.H + self.F, ), name='input_controls')

        self.encoder = self._build_latent_model(input_rays)

        if self.training_phase == 0:
            for layer in self.encoder.layers:
                layer.trainable = False

        self.z_mean, self.z_log_var = self.encoder(input_rays)

        self.gen_model = self._build_generative_model()

        outputs = []
        prev_y, pred_y = None, None
        for i in range(self.H + self.F - 1):
            if i >= self.H:
                prev_y = pred_y
            else:
                prev_y = Lambda(lambda x: x[:, i, :])(input_rays)
                if self.task_relevant:
                    prev_y = Lambda(self._get_task_relevant_feature)(prev_y)
            if self.training_phase == 0:
                z = Lambda(self._sampling_prior)([self.z_mean, self.z_log_var])
            else:
                z = Lambda(self._sampling)([self.z_mean, self.z_log_var])
            u = Lambda(lambda x: K.expand_dims(x[:, i], axis=-1))(input_controls)
            pred_y = self.gen_model([prev_y, u, z])
            outputs.append(pred_y)

        outputs_flat = outputs[0]
        for i in range(1, self.F + self.H - 1, 1):
            outputs_flat = Concatenate()([outputs_flat, outputs[i]])

        return Model([input_rays, input_controls], outputs_flat, name='vae_model')

    def load_weights(self, filename):
        self.training_phase = 2
        self.vae = self._build_vae_model()
        self.vae.load_weights(filename)
        self.vae.compile(optimizer='adam', loss=self._vae_loss_function)
        self.vae.summary()
        print("Loaded weights!")

    def train_model(self, x_train, x_val, y_train, y_val, u_train, u_val):
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
        self.vae = self._build_vae_model()
        if self.task_relevant:
            self.vae.load_weights("vae_weights_tr_p0.h5")
        else:
            self.vae.load_weights("vae_weights_p0.h5")
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

        self.training_phase = 2
        self.vae = self._build_vae_model()
        if self.task_relevant:
            self.vae.load_weights("vae_weights_tr_p1.h5")
        else:
            self.vae.load_weights("vae_weights_p1.h5")
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

    def get_gen_model(self):
        return self.gen_model

    def get_vae_model(self):
        return self.vae
