## Trajectory simulation
## Code to recreate the output from given samples:
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
from models import *
from costs import *
import os

## Initialize tensorflow session:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

## Reload the graph:
basefolder = 'Checkpoints/'
modelfolder = 'Video_ball_color_statvar_decoder_ckpts/'
data = 'modelep4999.ckpt'
metadata = data+'.meta'
saver = tf.train.import_meta_graph(basefolder+modelfolder+metadata)
saver.restore(sess,basefolder+modelfolder+data)

## Name vars to restore:
graph = tf.get_default_graph()

##
intensities = graph.get_tensor_by_name('intensities:0')
grountruth_images = graph.get_tensor_by_name('input_samples:0')
sampled_images = graph.get_tensor_by_name("deconv2d_4/Sigmoid:0")
# hidden_activations = graph.get_tensor_by_name('model_g/corr_samples_aug:0')
#
# ## For the case of the static variable:
# mean_weights = graph.get_tensor_by_name('model_g/Stat_weights_mean:0')
# var_weights = graph.get_tensor_by_name('model_g/Stat_weights_var:0')
#
# params = sess.run([mean_weights,var_weights])
# print(params[0].shape)
# plt.plot(params[0][0,:])
# plt.savefig('means.png')
# plt.close()
# plt.plot(params[1][0,:])
# plt.savefig('vars.png')
# plt.close()
# #
# z_vmeans = np.load(basefolder+modelfolder+'means.npy')
# z_vvars = np.load(basefolder+modelfolder+'vars.npy')
# print(z_vmeans.shape)
# fig,ax = plt.subplots(2,1)
# ax[0].plot(np.linspace(0,1000,100),z_vmeans[:,0,0],'bo')
# ax[0].plot(np.linspace(0,1000,100),z_vmeans[:,0,0]+z_vvars[:,0,0],'r--')
# ax[0].plot(np.linspace(0,1000,100),z_vmeans[:,0,0]-z_vvars[:,0,0],'r--')
# ax[1].plot(np.linspace(0,1000,100),z_vmeans[:,0,1],'bo')
# ax[1].plot(np.linspace(0,1000,100),z_vmeans[:,0,1]+z_vvars[:,0,1],'r--')
# ax[1].plot(np.linspace(0,1000,100),z_vmeans[:,0,1]-z_vvars[:,0,1],'r--')
# # ax[0].axvline(x = 81)
# # ax[1].axvline(x = 81)
# # ax[0].axvline(x = 296)
# # ax[1].axvline(x = 296)
# plt.savefig('z_evolution.png')
# plt.close()



## Import the samples that were generated:
# samples = np.load(basefolder+modelfolder+'samples.npy')

# plt.plot(samples[:,0],samples[:,1],'--o')
# plt.savefig('latent_samples_big')
# plt.close()

# We replace the dynamical samples with their means:
# samples[:,0:2] = np.tile(np.mean(samples[:,0:2],axis = 0),(50,1))
# samples[:,2:] = np.zeros((50,2))#np.tile(np.mean(samples[:,0:2],axis = 0),(50,1))

# xmin = min(samples[:,0])
# xmax = max(samples[:,0])
# ymin = min(samples[:,1])
# ymax = max(samples[:,1])

## Generate input pipeline (unshuffled)
filenames = ['datadirectory/toydynamics_nograv/Video_ball_colorencoder_train.tfrecords']

imsize = 64
batch_size = 50
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

base_dataset = tf.data.TFRecordDataset(filenames)
# Get out the images and tags in a useful format
preprocessed = base_dataset.map(preprocess)
## We now want to batch the dataset into groups of x neighboring frames:
batches = preprocessed.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
## Make an iterator without shuffling
iterator = batches.make_initializable_iterator()
## now define the next set of images.
next_images_it,_,_,intensity_it = iterator.get_next()

# statvar_gt = tf.tile(tf.random_uniform([1,1],maxval = 1.),[batch_size,1])
#
# full_input = tf.concat((intensity_it,statvar_gt),axis = 1)
## Run initializer for your iterator
sess.run(iterator.initializer)
# input_int = sess.run(intensity_it)
# gen_images = sess.run(sampled_images,feed_dict={intensities:input_int})
# input_ims = sess.run(next_images_it)
# gen_images = sess.run(sampled_images,feed_dict={inputs:input_ims})

i = 0

while True:
    try:
        input_int = (sess.run(intensity_it)).reshape(batch_size,1)
        gen_images,orig_images = sess.run([sampled_images,next_images_it],feed_dict={intensities:input_int})
        # xmin = min(samples[:,0])
        # xmax = max(samples[:,0])
        # ymin = min(samples[:,1])
        # ymax = max(samples[:,1])
        for j in range(batch_size):
            fig,ax = plt.subplots(2,1)
            ax[0].imshow(gen_images[j,:,:,:])
            # ax[1].set_xlim((xmin-2,xmax+2))
            # ax[1].set_ylim((ymin-2,ymax+2))
            # ax[1].plot(samples[:j+1,0],samples[:j+1,1],'--o')
            ax[1].imshow(orig_images[j,:,:,:])
            plt.savefig('saveframe'+str(j+i*batch_size)+'.png')
            plt.close()
        # progress = i/(2400/(batch_size))
        # sys.stdout.write("Train progress: %d%%   \r" % (progress) )
        # sys.stdout.flush()
        # _,cost = sess.run([optimizer,total_cost])
        # epoch_cost+=cost
        i+=1
        print(i)
    except tf.errors.OutOfRangeError:
        break
# for i in range(batch_size):
#     plt.imshow(gen_images[i,:,:,:])
#     # ax[1].set_xlim((xmin-2,xmax+2))
#     # ax[1].set_ylim((ymin-2,ymax+2))
#     # ax[1].plot(samples[:i+1,0],samples[:i+1,1],'--o')
#
#     plt.savefig('saveframe'+str(i)+'.png')
#     plt.close()
subprocess.call(['ffmpeg', '-framerate', str(10), '-i', 'saveframe%01d.png', '-r', '30','Reconstruct_simtraj_'+modelfolder.split('/')[0]+'.mp4'])
