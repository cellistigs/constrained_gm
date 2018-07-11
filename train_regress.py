## Train a simple output regression network that takes in an intensity and
## a random static variable, and outputs an image that we're interested in.
## We can do this simply by repurposing the inputs that we have to our VIN, and
## Flipping the order in which they are implemented.

## Additionally, the cost is simple to implement as well.
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import *
from costs import *
import os
imsize = 64
## The datasets we use are altered versions of the full autoencoder datasets, that
## have as their output ground truth the amplitude of the cycle.
filenames = ['datadirectory/toydynamics_nograv/Video_ball_colorencoder_train.tfrecords']

def halfsize(image):
    return resize(image,(imsize,imsize))
### Make input pipeline:
# Define a function that wraps the preprocessing steps:
def preprocess(serialized_example):
    featureset = {'video': tf.FixedLenFeature([],tf.string),
                  'frame': tf.FixedLenFeature([],tf.int64),
                  'image': tf.FixedLenFeature([],tf.string),
                  'intensity': tf.FixedLenFeature([],tf.float32),
                  }
    features = tf.parse_single_example(serialized_example,features = featureset)

    # Convert string representation to integers:
    image = tf.decode_raw(features['image'],tf.float64)

    image = tf.reshape(image,[128,128,3])

    image = tf.image.resize_images(image,[imsize,imsize])

    image = tf.cast(image,tf.float32)
    # image.set_shape
    # Convert label to string:
    label = features['frame']
    video = features['video']
    intensity = features['intensity']
    return image,label,video,intensity

batch_size = 50

# Apply preprocessing
base_dataset = tf.data.TFRecordDataset(filenames)
# Get out the images and tags in a useful format
preprocessed = base_dataset.map(preprocess).shuffle(50)
## We now want to batch the dataset into groups of ten neighboring frames:
batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
## Each of these groups of 10 will be processed together, and their latents will
## reflect this structure.
## Shuffle data:
shuffled = batches.repeat(2).shuffle(40)
## Make an iterator:
iterator = shuffled.make_initializable_iterator()
## We will initialize and feed through this iterator at training time.
next_images_it,_,_,intensity_it = iterator.get_next()

## DESIGN THE INPUT
### This next line is to potentially make our lives easier when performing reconstructions.
next_images = tf.placeholder_with_default(next_images_it,shape = [None,imsize,imsize,3],name = 'input_samples')

intensity = tf.placeholder_with_default(tf.reshape(intensity_it,(batch_size,1)),shape = [None,1],name = 'intensities')

## We sample one instance of uniform noise that acts as our static objective:
statvar_gt = tf.tile(tf.random_uniform([1,1],maxval = 1.),[batch_size,1])

full_input = tf.concat((intensity,statvar_gt),axis = 1)

with pt.defaults_scope(activation_fn=tf.nn.elu,
                       learned_moments_update_rate = 0.0003,
                       variance_epsilon = 0.001,
                       scale_after_normalization=True):
    with pt.defaults_scope(phase=pt.Phase.train):
        gen_images = gener_model_dyn(full_input)

## Now design a cost. This is just  the gaussian nll without regularization.
## We are just doing least squares on it.
resid = regression_cost(next_images,gen_images,batch_size)

optimizer = tf.train.AdamOptimizer(5e-5,epsilon=1e-10).minimize(resid)

print('Running Tensorflow model')
checkpointdirectory = 'Video_ball_color_statvar_decoder_ckpts'
init = tf.global_variables_initializer()
if not os.path.exists(checkpointdirectory):
    os.mkdir(checkpointdirectory)

# #Add iterator to the list of saveable objects:
#
# saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
#
# # Save the iterator state by adding it to the saveable objects collection.
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(init)
    allmeans = []
    allvars = []
    max_epochs = 5000
    for epoch in range(max_epochs):
        sess.run(iterator.initializer)
        # epoch_loss = 0
        epoch_cost = 0
        i = 0
        while True:
            try:
                progress = i/(2400/(batch_size))*100
                sys.stdout.write("Train progress: %d%%   \r" % (progress) )
                sys.stdout.flush()
                _,cost = sess.run([optimizer,resid])
                epoch_cost+=cost
                i+=1
            except tf.errors.OutOfRangeError:
                break
        print('Loss for epoch '+str(epoch)+': '+str(epoch_cost))
        save_path = saver.save(sess,checkpointdirectory+'/modelep'+str(epoch)+'.ckpt')
