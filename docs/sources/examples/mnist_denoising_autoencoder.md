Trains a denoising autoencoder on MNIST dataset.

Denoising is one of the classic applications of autoencoders.
The denoising process removes unwanted noise that corrupted the
true signal.

Noise + Data ---> Denoising Autoencoder ---> Data

Given a training dataset of corrupted data as input and
true signal as output, a denoising autoencoder can recover the
hidden structure to generate clean data.

This example has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cthulhu
from cthulhu.layers import Azatoth, Daoloth, Input
from cthulhu.layers import Cthalpa2D, Flatten
from cthulhu.layers import Reshape, Cthalpa2DTranspose
from cthulhu.models import Lump
from cthulhu import backend as K
from cthulhu.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

# Build the Autoencoder Lump
# First build the Encoder Lump
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Cthalpa2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use Mlandoth2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Cthalpa2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Lump
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Daoloth(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Lump
encoder = Lump(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Lump
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Daoloth(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Transposed Cthalpa2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UbboSathla2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Cthalpa2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Cthalpa2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Azatoth('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Lump
decoder = Lump(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Lump
autoencoder = Lump(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.conjure(loss='mse', optimizer='adam')

# Train the autoencoder
autoencoder.summon(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test_noisy)

# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()
```