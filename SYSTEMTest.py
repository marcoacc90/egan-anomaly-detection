import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import Modules.Model as MO
import Modules.Global as GO
import os
import sys
import time

def main():
    PATCH_SIZE = GO.PATCH_SIZE
    latent_dim = GO.NOISE_DIM

    if len(sys.argv) != 8:
        print('python3.6 SYSTEMTest.py ... \n')
        sys.exit( 1 )

    print( 'model ' + str(sys.argv[ 1 ]))
    print( 'modelid ' + str(sys.argv[ 2 ]))
    print( 'dataset ' + str(sys.argv[ 3 ]))
    print( 'ganfolder ' + str(sys.argv[ 4 ]))
    print( 'encfolder ' + str(sys.argv[ 5 ]))
    print( 'ofolder ' + str(sys.argv[ 6 ]))
    print( 'oname ' + str(sys.argv[ 7 ]))

    model = str( sys.argv[ 1 ] )
    modelid = str( sys.argv[ 2 ] )
    dataset = str( sys.argv[ 3 ] )
    ganfolder = str( sys.argv[ 4 ] )
    encfolder = str( sys.argv[ 5 ] )
    ofolder = str( sys.argv[ 6 ] )
    oname = str( sys.argv[ 7 ] )

    # LOAD MODELS
    g_model = MO.make_generator_model( GO.IMAGE_DIM, latent_dim )
    g_model.load_weights( ganfolder + 'generator_weights_' + modelid )
    e_model = MO.make_encoder_model( GO.IMAGE_DIM, latent_dim )
    e_model.load_weights( encfolder + 'encoder_weights_' + modelid )
    if model == 'IZIf':
        d_model = MO.make_discriminator_model(GO.IMAGE_DIM )
        d_model.load_weights( ganfolder + 'discriminator_weights_' + modelid )
        d_features = tf.keras.Model( d_model.inputs, d_model.get_layer('flatten').output )

    os.system( 'mkdir ' + ofolder )
    ofile = ofolder + '/' +  oname
    os.system('ls ' + dataset + ' > Image.txt')
    f = open( ofile, "w" )
    with open('Image.txt', 'r') as filehandle:
        for line in filehandle:
            name = line[:-1]
            img = kimage.load_img(dataset + name)
            print( name )
            x = np.array( img )
            x = x.reshape( 1, PATCH_SIZE, PATCH_SIZE, 3 ).astype('float32')
            x = (x - 127.5) / 127.5
            start_time = time.time()
            Ex = e_model.predict( x )
            GEx = g_model.predict( Ex )
            if model == 'IZIf' :
                recon_features = d_features.predict( GEx )
                image_features = d_features.predict( x )
            if model == 'IZIf':
                loss = MO.mse( x[0,:,:,:], GEx[0,:,:,:] ) + MO.mse( image_features[0,:], recon_features[0,:] )
            else:
                loss = MO.mse( x[0,:,:,:], GEx[0,:,:,:] )
            time_sec = ( time.time() - start_time )
            f.write( '%f %f\n' % ( loss.numpy(), time_sec) )
    os.system('rm -r Image.txt')
    f.close()

if __name__ == "__main__":
    main()
