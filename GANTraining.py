import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import os
import sys

def summarize_performance( path, epoch, g_model, d_model, images, latent_dim, n_samples=100 ) :
    xReal, yReal = TO.generateRealSamples( images, n_samples )
    _, accReal = d_model.evaluate( xReal, yReal, verbose=0 )
    xFake, yFake = TO.generateFakeSamplesNormal( g_model, n_samples, latent_dim )
    _, accFake = d_model.evaluate(xFake, yFake, verbose=0 )
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (accReal*100, accFake*100))
    filename = '%s/generated_plot_e%03d.png' % ( path, epoch+1 )
    X = TO.conver2image( xFake )
    TO.savePlot( filename, X )
    filename = '%s/generator_weights_%03d' % (path,epoch + 1)
    g_model.save_weights(filename)
    filename = '%s/discriminator_weights_%03d' % (path,epoch + 1)
    d_model.save_weights(filename)

@tf.function
def train_step( g_model, d_model, g_opt, d_opt, images, latent_dim ):
	noise = tf.random.normal([len(images), latent_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = g_model(noise, training=True)
		real_output = d_model(images, training=True)
		fake_output = d_model(generated_images, training=True)
		gen_loss = MO.generator_loss(fake_output)
		disc_loss = MO.discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

	g_opt.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
	d_opt.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))

def train( path, g_model, d_model, g_opt, d_opt, dataset, train_images, epochs, latent_dim ):
  for epoch in range(epochs):

    start = time.time()
    for image_batch in dataset:
      train_step( g_model, d_model, g_opt, d_opt, image_batch, latent_dim )
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    if (epoch + 1) % 10 == 0:
        summarize_performance( path, epoch, g_model, d_model, train_images, latent_dim )


# LOAD DATA FOR TRAINING
if len(sys.argv) != 2:
    print('python3.6 training.py <img_dir> \n')
    print('python3.6 training.py Dataset/normalTraining/');
    sys.exit( 1 )

# LOAD IMAGES AND CONSTRUCT THE PATCHES
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
os.system('rm -r Image.txt')
print('Patches are ready, shape: {}'.format(train_images.shape))
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize the images to [-1, 1]

BUFFER_SIZE = len(train_images)
BATCH_SIZE = GO.BATCH_SIZE
noise_dim = GO.NOISE_DIM
EPOCHS = GO.N_EPOCHS

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# NAME OF THE OUTPUT PATH
path = 'E' + str(EPOCHS) +'_GAN/'
cmd = 'mkdir ' + path
os.system( cmd )

# CREATE THE MODELS AND OPTIMIZERS
g_model = MO.make_generator_model( noise_dim )
d_model = MO.make_discriminator_model()
g_opt = tf.keras.optimizers.Adam( 1e-4 )
d_opt = tf.keras.optimizers.Adam( 1e-4 )

# START THE TRAINING
train( path, g_model, d_model, g_opt, d_opt, dataset, train_images, EPOCHS, noise_dim )
