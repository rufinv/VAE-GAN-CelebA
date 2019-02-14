"""Build the VAEGAN network

Part of the code was adapted from:
Parag K. Mital
Creative applications of deep learning
(Kadenze course)

"""

import tensorflow as tf
import libs.utils as utils
import numpy as np

def encoder(x, is_training, channels, filter_sizes, activation=tf.nn.tanh, reuse=None):
    # Set the input to a common variable name, h, for hidden layer
    h = x

    print('encoder/input:', h.get_shape().as_list())
    # Now we'll loop over the list of dimensions defining the number
    # of output filters in each layer, and collect each hidden layer
    hs = []
    for layer_i in range(len(channels)):
        
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            # Convolve using the utility convolution function
            # This requirs the number of output filter,
            # and the size of the kernel in `k_h` and `k_w`.
            # By default, this will use a stride of 2, meaning
            # each new layer will be downsampled by 2.
            h, W = utils.conv2d(h, channels[layer_i],
                                k_h=filter_sizes[layer_i],
                                k_w=filter_sizes[layer_i],
                                d_h=2,
                                d_w=2,
                                reuse=reuse)
            
            h = utils.batch_norm(h, is_training)

            # Now apply the activation function
            h = activation(h)
            print('layer:', layer_i, ', shape:', h.get_shape().as_list())
            
            # Store each hidden layer
            hs.append(h)

    # Finally, return the encoding.
    return h, hs
    
def variational_bayes(h, n_code):
    # Model mu and log(\sigma)
    z_mu = tf.nn.tanh(utils.linear(h, n_code, name='mu')[0])
    z_log_sigma = 0.5 * tf.nn.tanh(utils.linear(h, n_code, name='log_sigma')[0])

    # Sample from noise distribution p(eps) ~ N(0, 1)
    epsilon = tf.random_normal(tf.stack([tf.shape(h)[0], n_code]))

    # Sample from posterior
    z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))

    # Measure loss
    loss_z = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)

    return z, z_mu, z_log_sigma, loss_z

def decoder(z, is_training, dimensions, channels, filter_sizes,
            activation=tf.nn.elu, reuse=None):
    h = z
    for layer_i in range(len(dimensions)):
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            h, W = utils.deconv2d(x=h,
                               n_output_h=dimensions[layer_i],
                               n_output_w=dimensions[layer_i],
                               n_output_ch=channels[layer_i],
                               k_h=filter_sizes[layer_i],
                               k_w=filter_sizes[layer_i],
                               reuse=reuse)
            h = utils.batch_norm(h, is_training)
            h = activation(h)
    return h


def make_network():
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='istraining')

    n_pixels = 128
    n_channels = 3
    input_shape = [None, n_pixels, n_pixels, n_channels]

    # placeholder for the input to the network
    X = tf.placeholder(name='X',dtype = np.float32,shape=input_shape)

    channels = [192, 256, 384, 512, 768]
    filter_sizes = [3, 3, 3, 3, 3]
    activation = tf.nn.elu
    n_hidden = 1024

    with tf.variable_scope('encoder'):
        H, Hs = encoder(X, is_training, channels, filter_sizes, activation)
        Z = utils.linear(H, n_hidden)[0]

    n_code = 1024

    with tf.variable_scope('encoder/variational'):
        Z, Z_mu, Z_log_sigma, loss_Z = variational_bayes(h=Z, n_code=n_code)

    dimensions = [n_pixels // 16, n_pixels // 8, n_pixels // 4, n_pixels // 2, n_pixels]
    channels = [512, 384, 256, 192, n_channels]
    filter_sizes = [3, 3, 3, 3, 3, 3]
    activation = tf.nn.elu
    n_latent = n_code * (n_pixels // 32)**2

    with tf.variable_scope('generator'):
        Z_decode = utils.linear(
            Z, n_output=n_latent, name='fc', activation=activation)[0]
        Z_decode_tensor = tf.reshape(
            Z_decode, [-1, n_pixels//32, n_pixels//32, n_code], name='reshape')
        G = decoder(
            Z_decode_tensor, is_training, dimensions,
            channels, filter_sizes, activation)
    
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
        
    return sess, X, G, Z, Z_mu, is_training, saver
