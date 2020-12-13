from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()

def make_generator_model(my_shape, latent_dim):
    model = tf.keras.Sequential()
    path_w, path_h, channel = my_shape
    ksize = path_w // 4
    model.add(layers.Dense( ksize * ksize * 256, use_bias=False, input_dim = latent_dim ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape(( ksize, ksize, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) # (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)) #(None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) #(None, 28, 28, 3)
    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def make_discriminator_model( my_shape ):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=my_shape, name='conv1'))
    model.add(layers.LeakyReLU( name = 'leaky1'))
    model.add(layers.Dropout(0.3, name = 'dropout1'))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', name = 'conv2'))
    model.add(layers.LeakyReLU(name = 'leaky2'))
    model.add(layers.Dropout(0.3, name = 'dropout2'))

    model.add(layers.Flatten(name = 'flatten'))
    model.add(layers.Dense(1, name = 'dense'))

    opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer= opt, loss='binary_crossentropy',metrics=['accuracy'])
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def make_encoder_model( my_shape, latent_dim ):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=my_shape)) #1, 14x14, 12x12, 16x16
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False )) #64, 7x7, 6x6, 8x8
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)) #128, 7x7, 6x6, 8x8
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense( latent_dim, activation='tanh'))

    opt = tf.keras.optimizers.RMSprop(1e-3)
    model.compile(optimizer= opt, loss='mean_squared_error')
    return model

def encoder_loss( real_images, rec_images ):
    return mse( real_images, rec_images )
