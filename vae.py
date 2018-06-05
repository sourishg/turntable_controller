from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Constants and params
theta_range = np.deg2rad(120.0)
num_rays = 100

H = 9 # no of past observations
F = 1 # no of future predictions

training_data_fraction = 0.8

def prepareDataset(filename):
  with open(filename) as f:
    x_raw = []
    y_raw = []
    lines = [line.rstrip('\n') for line in f]
    for i in range(len(lines)-H-F):
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
    n_test_samples = n - n_train_samples

    training_idx = np.random.randint(x.shape[0], size=n_train_samples)
    test_idx = np.random.randint(x.shape[0], size=n_test_samples)
    
    x_train, x_test = x[training_idx,:], x[test_idx,:]
    y_train, y_test = y[training_idx,:], y[test_idx,:]

    print("Prepared dataset!")
    return x_train, y_train, x_test, y_test

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

def plot_results(model,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
  #encoder, decoder = models
  x_test, y_test = data

  print "xtest", x_test[0]

  #z_mean, z_log_var, z = encoder.predict(x_test)
  
  #print "latent", z[0]


  y_pred = model.predict(x_test)

  print "y_pred", y_pred[0]

  vae.evaluate(x_test, y_test, batch_size=batch_size)

  theta_inc = theta_range / float(num_rays)
  plt.plot([i * np.rad2deg(theta_inc) for i in range(num_rays)], y_test[0], 'r.')
  plt.plot([i * np.rad2deg(theta_inc) for i in range(num_rays)], y_pred[0], 'b.')
  plt.show()

x_train, y_train, x_test, y_test = prepareDataset(sys.argv[1])

# network parameters
original_dim = x_train.shape[1]
output_dim = y_train.shape[1]

input_shape = (original_dim, )
output_shape = (output_dim, )
intermediate_dim = 505
batch_size = 128
latent_dim = 101
epochs = 5

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
y = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(output_dim, activation='sigmoid')(y)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

def vae_loss_function(y_true, y_pred):
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  mse_loss = mse(y_true, y_pred)
  return K.mean(mse_loss + kl_loss)

if __name__ == '__main__':
  trained = False
  data = (x_train, y_train)

  vae.compile(optimizer='adam', loss=vae_loss_function)
  vae.summary()
  
  if trained:
    # load weights into new model
    vae.load_weights("vae_weights.h5")
    vae.compile(optimizer='adam', loss=vae_loss_function)
    print("Loaded model from disk")
  else:
    vae.fit(x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test))
    # serialize weights to HDF5
    vae.save_weights("vae_weights.h5")
    print("Saved model to disk")

  plot_model(vae, to_file='vae.png', show_shapes=True)
  plot_results(vae, data, batch_size=batch_size, model_name="vae")
