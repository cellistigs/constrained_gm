## Define the architectures used in the constrained generative model setting.
import sys
import tensorflow as tf
import prettytensor as pt
from deconv import deconv2d

# Global network parameters

def recog_model_dyn(input_tensor,dim_z,dim_x):
    ''' The input to this network (input_tensor) is a set of batches, each of which has
    temporal ordering. i.e. the examples in a single batch follow temporal dynamics
    We do this because we want to take the batched output, and apply dynamical transformations
    to it in order to generate samples from the appropriate approximate posterior.
    We assume that it comes as a 2d tensor, as [batch, everything]
    '''
    ## We assume we have a latent system with dimension 2


    ## Construct the shared layers of the network
    shared_layers = (pt.wrap(input_tensor).
                     reshape([None, dim_x, dim_x, 3]). ## Reshape input
                     conv2d(5, 16, stride=2). ## Three layers of convolution: 32x32x4
                     conv2d(5, 32, stride=2). ## 16x16x8
                     conv2d(5, 64, stride=2). ## 8x8x16
                     conv2d(5, 128, stride=2). ## 4x4x32
                     conv2d(5, 256, stride=2). ## 2x2x
                     dropout(0.9).
                     flatten())

    mean_output = shared_layers.fully_connected(dim_z,activation_fn=None,name = 'means',weights=tf.random_uniform_initializer(0.1)).tensor
    covar_output = shared_layers.fully_connected(dim_z*dim_z,activation_fn=tf.nn.tanh,name = 'covars',weights=tf.random_uniform_initializer(0.1)).tensor

    return mean_output,covar_output

def gener_model_dyn(hidden_activations):
    '''The input to this network (hidden_activations) is a set of sampled activations that
    represents hidden activations across a batch. They are correlated by means of the structure imposed
    on the noise that they experience when they are sampled together, but here they should be shaped in a
    batch-friendly setup, that is once again a 2d tensor: [batch,everything]. We will first connect to a
    fully connected layer in order to generate input that is correctly shaped for deconvolution that mirrors
    the recognition model.
    '''
    return (pt.wrap(hidden_activations).
            fully_connected(2*2*256,activation_fn=None,weights=tf.random_uniform_initializer(0.1)).
            reshape([None,2,2,256]).
            deconv2d(5,128,stride = 2).
            deconv2d(5,64,stride = 2).
            deconv2d(5,32,stride = 2).
            deconv2d(5,16,stride = 2).
            deconv2d(5,3,stride = 2,activation_fn=tf.nn.sigmoid)).tensor # 32x32x4
