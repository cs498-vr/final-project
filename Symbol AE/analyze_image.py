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

image_size = 28

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
