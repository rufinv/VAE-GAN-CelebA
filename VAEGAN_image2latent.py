#!/usr/bin/python3

# First check the Python version
import sys, getopt
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n',
          'You should consider updating to Python 3.4.0 or',
          'higher.\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import libs.make_network as make_network
    import libs.utils as utils
    from scipy.io import loadmat, savemat
except ImportError as e:
    print("Make sure the libs folder is available in current directory.")
    print(e)

print('TF version = ',tf.__version__)

sys.path

def main(argv):
    file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["help","ifile="])
    except getopt.GetoptError:
        print ('VAEGAN_image2latent.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('VAEGAN_image2latent.py -i <inputfile>')
            print ('   Will load the vaegan_celeba.ckpt model (make sure it''s in the folder),')
            print ('   apply it to the input image (only one image at a time) to compute the')
            print ('   1024 latent variables, saved as inputfile_z.mat (Matlab format).')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file = arg
            print ('Input file is ', file)

    sess, X, G, Z, Z_mu, is_training, saver = make_network.make_network()
    if os.path.exists("vaegan_celeba.ckpt"):
        saver.restore(sess, "vaegan_celeba.ckpt")
        print("VAE-GAN model restored.")
    else:
        print("Pre-trained network appears to be missing.")
        sys.exit()

    img = plt.imread(file)[..., :3]
    img = utils.preprocess128(img,crop_factor=0.8)[np.newaxis]
                   
    #generate images from z
    z = sess.run(Z_mu, feed_dict={X: img, is_training: False})

    #save data in Matlab format
    savemat(file[:-4]+'_z',dict(latent=z))
                   

if __name__ == "__main__":
   main(sys.argv[1:])


