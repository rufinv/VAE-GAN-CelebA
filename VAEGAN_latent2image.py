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
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data, io
    from scipy.misc import imresize
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.interpolation import rotate
    import IPython.display as ipyd
    import tensorflow as tf
    from libs import utils, make_network
    from libs.batch_norm import batch_norm
except ImportError as e:
    print("Make sure the libs folder is available in current directory.")
    print(e)

print('TF version = ',tf.__version__)


def main(argv):
    file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print ('VAEGAN_latent2im.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == ('-h', '--help':
            print ('VAEGAN_latent2im.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file = arg
            print ('Input file is "', file)

    sess, X, G, Z, is_training, saver = make_network()
    if os.path.exists("vaegan_celeba.ckpt"):
        saver.restore(sess, "vaegan_celeba.ckpt")
        print("VAE-GAN model restored.")
    else:
        print("Pre-trained network appears to be missing.")


    #load data from Matlab format
    from scipy.io import loadmat, savemat
    pred = loadmat(file)['predictions']
    print('input data to be transformed:',pred.shape)

    #generate images from z
    g = sess.run(G, feed_dict={Z: (1*pred), is_training: False})

    fig, axs = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(20):
        axs[i//5,i%5].imshow(imdeprocess(g[i][np.newaxis])), axs[i//5,i%5].grid('off'), axs[i//5,i%5].axis('off')
    

    fig.savefig(file[:-4]+'.jpg')
    savemat(file[:-4]+'_images',dict(images=g))


if __name__ == "__main__":
   main(sys.argv[1:])


