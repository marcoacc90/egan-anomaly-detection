from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()

def make_generator_model(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_dim = latent_dim ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) # (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)) #(None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) #(None, 28, 28, 3)
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 3], name='conv1'))
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

def make_encoder_model( latent_dim ):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 3])) #1, 14x14
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False )) #64, 28x28
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)) #128, 28x28
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Flatten())
	model.add(layers.Dense( latent_dim, activation='tanh'))
	opt = tf.keras.optimizers.RMSprop(1e-3)
	model.compile(optimizer= opt, loss='mean_squared_error')
	return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def encoder_loss( real_images, rec_images ):
    return mse( real_images, rec_images )
