import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import os
import sys
import time

if len(sys.argv) != 6:
    print('python3.6 AUTOENCODERTest.py ... \n')
    sys.exit( 1 )

# LOAD MODELS
PATCH_SIZE = GO.PATCH_SIZE
latent_dim = GO.NOISE_DIM

print( 'modelid ' + str(sys.argv[ 1 ]))
print( 'dataset ' + str(sys.argv[ 2 ]))
print( 'autoencfolder ' + str(sys.argv[ 3 ]))
print( 'ofolder ' + str(sys.argv[ 4 ]))
print( 'oname ' + str(sys.argv[ 5 ]))

modelid = str( sys.argv[ 1 ] )
dataset = str( sys.argv[ 2 ] )
autoencfolder = str( sys.argv[ 3 ] )
ofolder = str( sys.argv[ 4 ] )
oname = str( sys.argv[ 5 ] )

# LOAD MODELS
encoder = MO.make_encoder_model( latent_dim )
decoder = MO.make_generator_model( latent_dim )
autoencoder = tf.keras.Sequential( [ encoder, decoder ] )
autoencoder.load_weights( autoencfolder + 'autoencoder_weights_' + modelid )

# TEST
os.system( 'mkdir ' + ofolder )
ofile = ofolder + '/' +  oname
os.system('ls ' + dataset + ' > Image.txt')
f = open( ofile, "w" )
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(dataset + name )
        print( name )
        x = np.array( img )
        x = x.reshape( 1, PATCH_SIZE, PATCH_SIZE, 3).astype('float32')
        x = (x - 127.5) / 127.5
        start_time = time.time()
        x_est = autoencoder.predict( x )
        loss = MO.mse( x[0,:,:,:], x_est[0,:,:,:] )
        time_sec = ( time.time() - start_time )
        f.write( '%f %f\n' % ( loss.numpy(), time_sec) )
os.system('rm -r Image.txt')
f.close()
