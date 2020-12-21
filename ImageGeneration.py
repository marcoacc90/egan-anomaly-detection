import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import os
import sys

# Generate the reconstructed images by
    # AUTOENCODER
    # GANIZIf
PATCH_SIZE = GO.PATCH_SIZE
latent_dim = GO.NOISE_DIM

def loadDataset( dataset ) :
    count = 0
    os.system('ls ' + dataset + ' > Image.txt')
    flag = False
    with open('Image.txt', 'r') as filehandle:
        for line in filehandle:
            name = line[:-1]
            img = kimage.load_img(dataset + name)
            x = np.array( img )
            x = x.reshape( (1,PATCH_SIZE,PATCH_SIZE,3) )
            if flag == True:
                test_images = np.concatenate( (test_images,x), axis = 0)
            else:
                test_images = np.copy( x )
                flag = True
            count = count + 1
            if count == 1000 :
                break;
    os.system('rm -r Image.txt')
    test_images = test_images.astype('float32')
    test_images = (test_images - 127.5) / 127.5
    return test_images

def main():
    modelid = str( 500 )
    ofolder = 'Reconstruction/'
    dataset = 'Dataset2' #'Dataset1'

    #aefolder = 'E' + modelid + '_AE_'
    iziffolder = 'E' + modelid + '_IZIf_' + dataset #dataset +'Models/'+ 'E' + modelid + '_IZIf_P32_L100/'
    ganfolder = 'E' + modelid + '_GAN_' + dataset #dataset + 'Models/' + 'E' + modelid + '_GAN_P32_L100/'

    # LOAD MODELS
    e_model = MO.make_encoder_model( GO.IMAGE_DIM, latent_dim )
    g_model = MO.make_generator_model( GO.IMAGE_DIM, latent_dim )

    #e_model_ae = MO.make_encoder_model(  latent_dim )
    #g_model_ae = MO.make_generator_model( latent_dim )
    #autoencoder = tf.keras.Sequential( [ e_model_ae, g_model_ae ] )

    g_model.load_weights( ganfolder  + '/generator_weights_' + modelid )
    e_model.load_weights( iziffolder  + '/encoder_weights_' + modelid )
    #autoencoder.load_weights( aefolder +'autoencoder_weights_' + modelid )

    # CREATE FOLDERS
    os.system( 'mkdir ' + ofolder )

    # LOAD DATASET
    x_normal = loadDataset( dataset + '/normalTest32/' )
    x_novel = loadDataset( dataset  + '/novelTest32/' )

    # SAVE
    nsol = 10
    n_samples = 100
    for i in range( nsol ) :
        x,_ = TO.generateRealSamples( x_normal, n_samples )
        y,_ = TO.generateRealSamples( x_novel, n_samples )

        #xAE = autoencoder.predict( x )
        z = e_model.predict( x )
        xIZIf = g_model.predict( z )

        #yAE = autoencoder.predict( y )
        z = e_model.predict( y )
        yIZIf = g_model.predict( z )

        # Real images
        x = TO.conver2image( x )
        TO.savePlotRectangle( ofolder + str(i) + '_normal.pdf', x, n=20, m=4)
        y = TO.conver2image( y )
        TO.savePlotRectangle( ofolder + str(i) + '_novel.pdf', y, n=20, m=4)

        # Fake images
        xFake,_ = TO.generateFakeSamplesNormal( g_model, n_samples, 100 )
        xFake = TO.conver2image( xFake )
        TO.savePlotRectangle( ofolder + str(i) + '_gan.pdf', xFake, n=20, m=4)

        # Reconstructed
        '''
        xAE = TO.conver2image( xAE )
        TO.savePlotRectangle( ofolder + str(i) + '_normal_' + 'ae' + '.pdf', xAE )
        yAE = TO.conver2image( yAE )
        TO.savePlotRectangle( ofolder + str(i) + '_novel_' + 'ae' + '.pdf', yAE )
        '''
        xIZIf = TO.conver2image( xIZIf )
        TO.savePlotRectangle( ofolder + str(i) + '_normal_' + 'izif' + '.pdf', xIZIf, n=20, m=4 )
        yIZIf = TO.conver2image( yIZIf )
        TO.savePlotRectangle( ofolder + str(i) + '_novel_' + 'izif' + '.pdf', yIZIf, n=20, m=4)

if __name__ == "__main__":
    main()
