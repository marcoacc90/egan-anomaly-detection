import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import numpy as np
import os
import sys
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO

def summarize_performance( path, epoch, e_model, g_model, d_features, images, n_samples=100 ) :
	img,_ = TO.generateRealSamples( images, n_samples )
	z = e_model.predict( img )
	recon_images = g_model.predict( z )
	recon_features = d_features.predict( recon_images )
	image_features = d_features.predict( img )
	loss_img = MO.mse( img, recon_images )
	loss_fts = MO.mse( image_features, recon_features )
	enc_loss = loss_img + loss_fts

	print( 'Epoch: %d ' %(epoch+1) )
	print( enc_loss )
	filename = '%s/plot_e%03d_recon.png' % ( path, epoch+1 )
	X = TO.conver2image( recon_images )
	TO.savePlot( filename, X )
	filename = '%s/plot_e%03d_img.png' % ( path, epoch+1 )
	Y = TO.conver2image( img )
	TO.savePlot( filename, Y )
	filename = '%s/encoder_weights_%03d' % (path,epoch + 1)
	e_model.save_weights( filename )

@tf.function
def train_step( e_model, g_model, d_features, e_opt, images ):
	with tf.GradientTape() as enc_tape:
		z = e_model( images, training = True )
		recon_images = g_model( z, training = False )

		recon_features = d_features( recon_images, training = False )
		image_features = d_features( images, training = False )

		loss_img = MO.mse( images, recon_images )
		loss_fts = MO.mse( image_features, recon_features )
		enc_loss = loss_img + loss_fts

	gradients_of_encoder = enc_tape.gradient( enc_loss, e_model.trainable_variables)
	e_opt.apply_gradients(zip(gradients_of_encoder, e_model.trainable_variables))

def train( path, e_model, g_model, d_features, e_opt, dataset, train_images, epochs, latent_dim ):
  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      train_step( e_model, g_model, d_features, e_opt, image_batch)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    if (epoch + 1) % 100 == 0:
        summarize_performance( path, epoch, e_model, g_model, d_features, train_images )

# LOAD DATA FOR TRAINING
if len(sys.argv) != 4:
    print('python3.6 ENCODERTrainingIZIf.py <dataset> <ganfolder> <iziffolder> \n')
    print('python3.6 ENCODERTrainingIZIf.py ...');
    sys.exit( 1 )

PATCH_SIZE = GO.PATCH_SIZE
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
path = str(sys.argv[ 3 ])
cmd = 'mkdir ' + path
os.system( cmd )

# CREATE AND LOAD THE GENERATOR MODEL
name = str(sys.argv[ 2 ]) + 'generator_weights_' + '%03d' % (EPOCHS)
g_model = MO.make_generator_model(GO.IMAGE_DIM, noise_dim )
g_model.load_weights( name )

# CREATE AND LOAD THE DISCRIMINATOR MODEL
name = str(sys.argv[ 2 ]) +'discriminator_weights_' + '%03d' % (EPOCHS)
d_model = MO.make_discriminator_model( GO.IMAGE_DIM )
d_model.load_weights( name )
d_features = tf.keras.Model( d_model.inputs, d_model.get_layer('flatten').output ) # Create a model without the classification layer

e_model = MO.make_encoder_model( GO.IMAGE_DIM, noise_dim )
e_opt = tf.keras.optimizers.RMSprop( 1e-3 )

# START THE TRAINING
train( path, e_model, g_model, d_features, e_opt, dataset, train_images, EPOCHS, noise_dim )
