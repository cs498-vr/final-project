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

latent_dim = 2
image_size = 28

if len(sys.argv)-1 != latent_dim:
    print("Invalid Arguments: Expected %d floats." % latent_dim,
            "Got %d arguments." % (len(sys.argv)-1))
    z_sample = np.zeros((1, latent_dim))
else:
    z_sample = np.array([sys.argv[1:]], dtype=float)

print("Using", z_sample[0])

decoder = load_model("decoder.h5")

x_hat = decoder.predict(z_sample)
digit = x_hat[0].reshape(image_size, image_size)

plt.figure(figsize=(10, 10))
plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.imshow(digit, cmap='Greys')
plt.savefig("output.png")
plt.show()
