import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import numpy as np
import os
import sys
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import math

def summarize_performance( path, epoch, e_model, g_model, noise_dim, n_samples=100 ) :
	noise = tf.random.normal([n_samples, noise_dim])
	images = g_model.predict( noise )
	z_est = e_model.predict( images )
	enc_loss = MO.encoder_loss( noise, z_est )
	print( 'Epoch: %d ' %(epoch+1) )
	print( enc_loss )
	filename = '%s/encoder_weights_%03d' % (path,epoch + 1)
	e_model.save_weights(filename)

@tf.function
def train_step( e_model, g_model, e_opt, noise_dim, batch_size ):
	noise = tf.random.normal([batch_size, noise_dim])

	with tf.GradientTape() as enc_tape:
		images = g_model( noise, training = False )
		z_est = e_model( images, training = True )
		enc_loss = MO.encoder_loss( noise, z_est )

	gradients_of_encoder = enc_tape.gradient( enc_loss, e_model.trainable_variables)
	e_opt.apply_gradients(zip(gradients_of_encoder, e_model.trainable_variables))

def train( path, e_model, g_model, e_opt, epochs, n_batches, batch_size, latent_dim ):
  for epoch in range(epochs):

    start = time.time()
    for i in range( n_batches ) :
      train_step( e_model, g_model, e_opt, latent_dim, batch_size )
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    if (epoch + 1) % 10 == 0:
        summarize_performance( path, epoch, e_model, g_model, latent_dim )


# LOAD DATA FOR TRAINING
if len(sys.argv) != 3:
    print('python3.6 ENCODERTrainingZIZ.py> GANFolder ZIZFolder\n')
    print('python3.6 ENCODERTrainingZIZ.py');
    sys.exit( 1 )

BUFFER_SIZE = 37800    # CURRENT ESTIMATION (DATASET2: 300*126)
BATCH_SIZE = GO.BATCH_SIZE
noise_dim = GO.NOISE_DIM
EPOCHS = GO.N_EPOCHS
N_BATCHES = math.ceil( BUFFER_SIZE / BATCH_SIZE )

# NAME OF THE OUTPUT PATH
path = str( sys.argv[ 2 ] )
cmd = 'mkdir ' + path
os.system( cmd )
name = str( sys.argv[ 1 ] ) + '/generator_weights_' + '%03d' % (EPOCHS)

# CREATE AND LOAD THE GENERATOR MODEL
g_model = MO.make_generator_model( noise_dim )
g_model.load_weights( name )
e_model = MO.make_encoder_model( noise_dim )
e_opt = tf.keras.optimizers.RMSprop( 1e-3 )

# START THE TRAINING
train( path, e_model, g_model, e_opt, EPOCHS, N_BATCHES, BATCH_SIZE, noise_dim )
