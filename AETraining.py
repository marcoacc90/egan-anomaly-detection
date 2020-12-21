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

# LOAD DATA FOR TRAINING
if len(sys.argv) != 2:
    print('python3 AUTOENCODERTraining.py <img_dir> \n')
    print('python3 AUTOENCODERTraining.py Dataset3/normalTraining/');
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
        x = x.reshape( (1, PATCH_SIZE, PATCH_SIZE,3) )
        if flag == True:
            train_images = np.concatenate( (train_images,x), axis = 0)
        else:
            train_images = np.copy( x )
            flag = True
os.system('rm -r Image.txt')
print('Patches are ready, shape: {}'.format(train_images.shape))
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = len(train_images)
BATCH_SIZE = GO.BATCH_SIZE
noise_dim = GO.NOISE_DIM
EPOCHS = GO.N_EPOCHS

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# NAME OF THE OUTPUT PATH
path = 'E' + str(EPOCHS) +'_AE'
cmd = 'mkdir ' + path
os.system( cmd )

# CREATE THE MODELS AND OPTIMIZERS
encoder = MO.make_encoder_model( GO.IMAGE_DIM, noise_dim )
decoder = MO.make_generator_model( GO.IMAGE_DIM, noise_dim )
autoencoder = tf.keras.Sequential( [ encoder, decoder ] )
autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop( 1e-3 ), loss='mean_squared_error' )

checkpoint_filepath = path + '/autoencoder_weights_{epoch:02d}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    period = 100,
    mode = 'max',
    save_best_only = False )

autoencoder.fit( train_images,
                 train_images,
                 batch_size = BATCH_SIZE,
                 shuffle = True,
                 epochs = EPOCHS,
                 callbacks = [model_checkpoint_callback]
                )
