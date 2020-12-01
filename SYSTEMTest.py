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

if len(sys.argv) != 1:
    print('python3.6 SYSTEMTest.py\n')
    print('python3.6 SYSTEMTest.py');
    sys.exit( 1 )

# LOAD MODELS
EPOCHS = GO.N_EPOCHS
MODEL = 'IZIf' #IZI, ZIZ, IZIf
mode = 'Validation'  # Validation,Test
latent_dim = GO.NOISE_DIM

dataset = 'Dataset/'
gan_folder = 'E' + str(EPOCHS) + '_GAN/'

path = 'E%d_Results' % (EPOCHS)
name = gan_folder + 'generator_weights_' + str(EPOCHS)
g_model = MO.make_generator_model( latent_dim )
g_model.load_weights( name )

if MODEL == 'IZIf':
	name =  gan_folder + 'discriminator_weights_' + '%03d' % (EPOCHS)
	d_model = MO.make_discriminator_model( )
	d_model.load_weights( name )
	d_features = tf.keras.Model( d_model.inputs, d_model.get_layer('flatten').output )
if MODEL == 'IZI':
	name = 'E500_ENC_%s/encoder_weights_%03d' %(MODEL,EPOCHS)
else :
	name = 'E%d_ENC_%s/encoder_weights_%03d' %(EPOCHS,MODEL,EPOCHS)
e_model = MO.make_encoder_model( latent_dim )
e_model.load_weights( name )

PATCH_SIZE = GO.PATCH_SIZE
N_SAMPLES = GO.N_SAMPLES

# TEST NORMAL
img_dir = 'Dataset2/normal' + mode + '/'
os.system('ls ' + img_dir + ' > Image.txt')
name = '%s/%s_loss_normal_%s.txt' % (path,MODEL,mode)
f = open( name, "w" )
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(img_dir + name)
        x = np.array( img )
        patch = simage.extract_patches_2d( x, (PATCH_SIZE,PATCH_SIZE), max_patches = N_SAMPLES )
        patch = patch.astype('float32')
        patch = (patch - 127.5) / 127.5
        z = e_model.predict( patch )
        patch_est = g_model.predict( z )
        if MODEL == 'IZIf' :
            recon_features = d_features.predict( patch_est )
            image_features = d_features.predict( patch )
        for i in range( len( patch ) ) :
            if MODEL == 'IZIf':
                loss = MO.mse( patch[i,:,:,:], patch_est[i,:,:,:]) + MO.mse( image_features[i,:], recon_features[i,:])
            else:
                loss = MO.mse( patch[i,:,:,:], patch_est[i,:,:,:] )
            f.write( '%f ' % ( loss.numpy() ) )
        f.write( '\n' )
os.system('rm -r Image.txt')
f.close()


# TEST ANOMALY
img_dir = 'Dataset2/novel' + mode + '/'
os.system('ls ' + img_dir + ' > Image.txt')
name = '%s/%s_loss_novel_%s.txt' % (path,MODEL,mode)
f = open( name, "w" )
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(img_dir + name)
        x = np.array( img )

        patch = simage.extract_patches_2d( x, (PATCH_SIZE,PATCH_SIZE), max_patches = N_SAMPLES )
        patch = patch.astype('float32')
        patch = (patch - 127.5) / 127.5


        z = e_model.predict( patch )
        patch_est = g_model.predict( z )
        if MODEL == 'IZIf' :
            recon_features = d_features.predict( patch_est)
            image_features = d_features.predict( patch )
        for i in range( len( patch ) ) :
            if MODEL == 'IZIf':
                loss = MO.mse( patch[i,:,:,:], patch_est[i,:,:,:]) + MO.mse( image_features[i,:], recon_features[i,:])
            else:
                loss = MO.mse( patch[i,:,:,:], patch_est[i,:,:,:] )
            f.write( '%f ' % ( loss.numpy() ) )
        f.write( '\n' )
os.system('rm -r Image.txt')
f.close()
