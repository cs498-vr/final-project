from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
# def sampling(args):
#     """Reparameterization trick by sampling from an isotropic unit Gaussian.
#     # Arguments
#         args (tensor): mean and log of variance of Q(z|X)
#     # Returns
#         z (tensor): sampled latent vector
#     """
#
#     z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
#     # by default, random_normal has mean = 0 and std = 1.0
#     epsilon = K.random_normal(shape=(batch, dim))
#     return z_mean + K.exp(0.5 * z_log_var) * epsilon
#
# def plot_results(models,
#                  data,
#                  batch_size=128,
#                  model_name="vae_mnist"):
#     """Plots labels and MNIST digits as a function of the 2D latent vector
#     # Arguments
#         models (tuple): encoder and decoder models
#         data (tuple): test data and label
#         batch_size (int): prediction batch size
#         model_name (string): which model is using this function
#     """
#
#     encoder, decoder = models
#     x_test, y_test = data
#     os.makedirs(model_name, exist_ok=True)
#
#     filename = os.path.join(model_name, "vae_mean.png")
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(x_test,
#                                    batch_size=batch_size)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig(filename)
#     # plt.show()
#     plt.savefig("latent_space.png")
#
#     filename = os.path.join(model_name, "digits_over_latent.png")
#     # display a 30x30 2D manifold of digits
#     n = 30
#     digit_size = 28
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
# #             z_sample = np.concatenate([np.array([[xi, yi]]), np.zeros((1,6))], axis=1)
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[i * digit_size: (i + 1) * digit_size,
#                    j * digit_size: (j + 1) * digit_size] = digit
#
#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig("digits.png")
#     # plt.savefig(filename)
#     # plt.show()
#
# image_size = 28
# original_dim = image_size * image_size
#
# # network parameters
# input_shape = (original_dim, )
# intermediate_dim = 512
# batch_size = 128
# latent_dim = 2
# epochs = 10
#
# # VAE model = encoder + decoder
# # build encoder model
# inputs = Input(shape=input_shape, name='encoder_input')
# x = Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = Dense(latent_dim, name='z_mean')(x)
# z_log_var = Dense(latent_dim, name='z_log_var')(x)
#
# # use reparameterization trick to push the sampling out as input
# # note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#
# # instantiate encoder model
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# # encoder.summary()
#
# # build decoder model
# latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# x = Dense(intermediate_dim, activation='relu')(latent_inputs)
# outputs = Dense(original_dim, activation='sigmoid')(x)
#
# # instantiate decoder model
# decoder = Model(latent_inputs, outputs, name='decoder')
# # decoder.summary()
#
# # instantiate VAE model
# outputs = decoder(encoder(inputs)[2])
# vae = Model(inputs, outputs, name='vae_mlp')
#
# models = (encoder, decoder)
#
# # VAE loss = mse_loss or xent_loss + kl_loss
# if True:
#     reconstruction_loss = mse(inputs, outputs)
# else:
#     reconstruction_loss = binary_crossentropy(inputs,
#                                               outputs)
#
# reconstruction_loss *= original_dim
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')
#
# # vae.load_weights('vae_mlp_mnist.h5')
# vae.load_weights('min_vae.h5')

if len(sys.argv)-1 != 1:
    print("Invalid Arguments: Expected [filename].",
            "Got %d arguments." % (len(sys.argv)-1))
    x = np.zeros(image_size * image_size);
else:
    with open(sys.argv[1], "r") as file:
        x = np.array(list(map(int, file.read().split()))).reshape(1, -1)

print("Using", x, x.shape)

encoder = load_model("encoder.h5")

z_hat = encoder.predict(x)[2][0]
print(z_hat)


# digit = x_hat[0].reshape(image_size, image_size)
#
# plt.figure(figsize=(10, 10))
# plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
# plt.imshow(digit, cmap='Greys')
# plt.savefig("output.png")
# plt.show()
