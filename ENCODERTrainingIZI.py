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

def summarize_performance( path, epoch, e_model, g_model, images, n_samples=100 ) :
	xReal,_ = TO.generateRealSamples( images, n_samples )
	zReal = e_model.predict( xReal )
	xRec = g_model.predict( zReal )
	enc_loss = MO.encoder_loss( xReal, xRec )
	print( 'Epoch: %d ' %(epoch+1) )
	print( enc_loss )
	filename = '%s/generated_plot_e%03d.png' % ( path, epoch+1 )
	X = TO.conver2image( xRec )
	TO.savePlot( filename, X )
	filename = '%s/encoder_weights_%03d' % (path,epoch + 1)
	e_model.save_weights(filename)

@tf.function
def train_step( e_model, g_model, e_opt, images):
	with tf.GradientTape() as enc_tape:
		z_real = e_model( images, training=True )
		rec_images = g_model( z_real, training=False)
		enc_loss = MO.encoder_loss( images, rec_images )

	gradients_of_encoder = enc_tape.gradient( enc_loss, e_model.trainable_variables)
	e_opt.apply_gradients(zip(gradients_of_encoder, e_model.trainable_variables))

def train( path, e_model, g_model, e_opt, dataset, train_images, epochs, latent_dim ):
  for epoch in range(epochs):

    start = time.time()
    for image_batch in dataset:
      train_step( e_model, g_model, e_opt, image_batch)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    if (epoch + 1) % 10 == 0:
        summarize_performance( path, epoch, e_model, g_model, train_images )


# LOAD DATA FOR TRAINING
if len(sys.argv) != 4:
    print('python3.6 ENCODERTrainingIZI.py <dataset> <ganfolder> <izifolder>\n')
    print('python3.6 ENCODERTrainingIZI.py ...');
    sys.exit( 1 )

PATCH_SIZE = GO.PATCH_SIZE
N_SAMPLES = GO.N_SAMPLES
img_dir = str( sys.argv[ 1 ] )
os.system('ls ' + img_dir + ' > Image.txt')
flag = False
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(img_dir + name)
        x = np.array( img )
        x = x.reshape( (1,PATCH_SIZE,PATCH_SIZE,3) )
        if flag == True:
            train_images = np.concatenate( (train_images,x), axis = 0)
        else:
            train_images = np.copy( x )
            flag = True
        print( train_images.shape )
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize the images to [-1, 1]
os.system('rm -r Image.txt')

BUFFER_SIZE = len(train_images)
BATCH_SIZE = GO.BATCH_SIZE
noise_dim = GO.NOISE_DIM
EPOCHS = GO.N_EPOCHS

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# NAME OF THE OUTPUT PATH
path = str( sys.argv[3] )
cmd = 'mkdir ' + path
os.system( cmd )
name = str( sys.argv[2] ) + 'generator_weights_' + '%03d' % (EPOCHS)

# CREATE AND LOAD THE GENERATOR MODEL
g_model = MO.make_generator_model( GO.IMAGE_DIM,  noise_dim )
g_model.load_weights( name )

e_model = MO.make_encoder_model( GO.IMAGE_DIM, noise_dim )
e_opt = tf.keras.optimizers.RMSprop( 1e-3 )

# START THE TRAINING
train( path, e_model, g_model, e_opt, dataset, train_images, EPOCHS, noise_dim )
