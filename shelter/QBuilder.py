# ref: http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
# ref: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
# ref: Playing Atari with Deep Reinforcement Learning

import tensorflow as tf

NACTIONS = 13
Nx, Ny, Nz, Nt = 80, 80, 1, 1

CH = Nz*Nt

def visualNetwork2D():
    """ Returns a shape of 1600, given [None, 80, 80, 1] """
    # Input and output classes
    x = tf.placeholder(tf.float32, shape=[None,Nx,Ny,1])
    y = tf.placeholder(tf.float32, shape=[None,NACTIONS])
    
    input_layer = tf.placeholder("float", [None, Nx, Ny, CH])
    
    # Layer shape is [None, 80, 80, 1]
    conv_w_1 = tf.Variable(tf.truncated_normal([7,7,1,32], stddev=0.01))
    conv_b_1 = tf.Variable(tf.truncated_normal([32]), stdev=0.01)
    conv1    = tf.nn.relu(tf.nn.conv2d(x, conv_w_1, strides=[1, 1, 1, 1], padding='SAME') + conv_b_1)
    
    max1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    # Layer shape is [None, 20, 20, 32]
    conv_w_2 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.01))
    conv_b_2 = tf.Variable(tf.truncated_normal([64]), stdev=0.01)
    conv2    = tf.nn.relu(tf.nn.conv2d(max1, conv_w_2, strides=[1, 1, 1, 1], padding='SAME') + conv_b_2)

    max2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    
    return tf.reshape(max2, [-1, 5*5*64]) # Layer shape [None, 5, 5, 64] 1600 Total
    
def qGenerator():
    """ Used as an iterative action selector in reinforcement learning.
        Takes a shape [-1, 1600] and returns a shape [-1, 13] """
    pass

def denseGenerator():
    """ Used as a creative generator.
        Takes a shape [None, Nx, Ny, Nz] and returns a shape [None, Nx, Ny, Nz]. """
    pass