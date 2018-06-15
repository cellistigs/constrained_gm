import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt
from models import *

## Script to train network.

## Load data:
filenames = ['datadirectory/toydynamics_nograv/Video_ball_colortrain.tfrecords']

### Make input pipeline:
# Define a function that wraps the preprocessing steps:
def preprocess(serialized_example):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = tf.cast(image,tf.float32)
    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    return image,label,video

## Helper function adapted from:
# https://stackoverflow.com/questions/42157781/block-diagonal-matrices-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
import tensorflow as tf

def block_diagonal(matrices, dtype=tf.float32):
  """Constructs block-diagonal matrices from a batch of 2D tensors.

  Args:
    matrices: A batch Tensor with shape [..., N_i, M_i]
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  # This adapted function assumes that the tensors are all the same shape and batched as such.
  ## We need to first figure out what the first dimension of the arrays looks like:
  shape = tf.shape(matrices)
  nb_tensors = shape[0]
  implicit_dim = shape[1]
  indices = tf.range(nb_tensors)
  # We now design an iterable function that will produce "padded" versions of each array that can then
  # be added together with a reduce_sum operation.
  max_zeros = implicit_dim*(nb_tensors-1)
  # this is ugly but okay
  map_padded = lambda args: tf.pad(args[0],tf.constant([[tf.random_normal((1)),5],[5,5]]))
  # map_padded = lambda args: (args[0]*tf.cast(args[1],tf.float32),args[1][0])

  # map_padded = lambda args: tf.pad(args[0],tf.constant([[max_zeros-implicit_dim*args[1],implicit_dim*args[1]],[max_zeros-implicit_dim*args[1],implicit_dim*args[1]]]))
  block_diag = tf.reduce_sum(tf.map_fn(map_padded,[matrices,indices],dtype = (tf.float32)),axis = 0)

  # blocked_rows = tf.Dimension(0)
  # blocked_cols = tf.Dimension(0)
  # batch_shape = tf.TensorShape(None)
  # for matrix in matrices:
  #   full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
  #   batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
  #   blocked_rows += full_matrix_shape[-2]
  #   blocked_cols += full_matrix_shape[-1]
  # ret_columns_list = []
  # for matrix in matrices:
  #   matrix_shape = tf.shape(matrix)
  #   ret_columns_list.append(matrix_shape[-1])
  # ret_columns = tf.add_n(ret_columns_list)
  # row_blocks = []
  # current_column = 0
  # for matrix in matrices:
  #   matrix_shape = tf.shape(matrix)
  #   row_before_length = current_column
  #   current_column += matrix_shape[-1]
  #   row_after_length = ret_columns - current_column
  #   row_blocks.append(tf.pad(
  #       tensor=matrix,
  #       paddings=tf.concat(
  #           [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
  #            [(row_before_length, row_after_length)]],
  #           axis=0)))
  # blocked = tf.concat(row_blocks, -2)
  # blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return block_diag,max_zeros

# Parameters:
dim_z = 2
batch_size = 10

print('Loading Data')
# Apply preprocessing
base_dataset = tf.data.TFRecordDataset(filenames)
# Get out the images and tags in a useful format
preprocessed = base_dataset.map(preprocess)
## We now want to batch the dataset into groups of ten neighboring frames:
batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
## Each of these groups of 10 will be processed together, and their latents will
## reflect this structure.
## Shuffle data:
shuffled = batches.repeat(2).shuffle(40)
## Make an iterator:
iterator = shuffled.make_initializable_iterator()
## We will initialize and feed through this iterator at training time.
next_images,_,_ = iterator.get_next()
### Now we define the flow of information through the network.

## Initialize dynamical parameters, A,Q_0,Q: (Copied from Evan Archer's theano code:
#  https://github.com/earcher/vilds/blob/master/code/tutorial.ipynb)

dyn_params = {'A':0.9*np.eye(dim_z),
              'Q_inv':np.eye(dim_z),
              'Q0_inv':np.eye(dim_z)}
print('Calculating Relevant Parameters')

# We can calculate D just once.
Qinv_full = np.kron(np.eye(batch_size),dyn_params['Q_inv'])
Qinv_full[0:dim_z,0:dim_z] = dyn_params['Q0_inv']
A_full = np.kron(np.eye(batch_size,k = -1),dyn_params['A'])
covar_prop = np.eye(batch_size*dim_z)-A_full

D_inv = np.matmul(np.matmul(covar_prop,Qinv_full),covar_prop.T)

print('Initializing Tensorflow Model')
### Movement through the network: Everything below this should be tensorflow ops!
## Run images through the recognition model:
with pt.defaults_scope(activation_fn=tf.nn.elu,
                       batch_normalize = True,
                       learned_moments_update_rate = 0.0003,
                       variance_epsilon = 0.001,
                       scale_after_normalization=True):
    with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope('model_g') as scope:
            mean_ind, rs = recog_model_dyn(next_images,dim_z = dim_z)

            ## Stochastic Layer
            # Reshape appropriately:
            mean_vec = tf.reshape(mean_ind,[batch_size*dim_z])
            rs_square= tf.reshape(rs,[batch_size,dim_z,dim_z])

            # Construct a covariance matrix by taking rs*rs_transpose
            covars = tf.matmul(rs_square,rs_square,transpose_b = True)
            print(covars.shape)
            C_inv = block_diagonal(covars)
            print(C_inv)
            # Now we can calculate the matrix square root through Cholesky decomposition,
            # and 'take an inverse' to find the true mean:
            # Sigma = (D_inv+C_inv)
            # R = tf.Cholesky(Sigma)
            #
            # mean_corr = tf.cholesky_solve(R,tf.matmul(C_inv,mean_vec))
            #
            # # Compute correlated noise as another inverse:
            # noise_corr = tf.matrix_triangular_solve(R,tf.random_normal([batch_size*dim_z]))
            #
            # samples = mean_corr+noise_corr
            #
            # ## Run samples through the generative model:
            # generated_images = gener_model_dyn(samples)

print('Running Tensorflow model')
init = tf.global_variables_initializer()

# saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)
    z = sess.run(C_inv)
    print(z,'!')
    while True:
        try:
            sess.run(generated_images)
        except tf.errors.OutOfRangeError:
            break








#
#
# next_images,next_label,_ = it.get_next()
