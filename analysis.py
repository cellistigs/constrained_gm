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
modelfolder = 'Video_ball_color_ckpts/'
data = 'modelep499.ckpt'
metadata = data+'.meta'
saver = tf.train.import_meta_graph(basefolder+modelfolder+metadata)
saver.restore(sess,basefolder+modelfolder+data)

## Name ops to restore:
graph = tf.get_default_graph()
sampled_images = graph.get_tensor_by_name("model_g/deconv2d_4/Sigmoid:0")
hidden_activations = graph.get_tensor_by_name('model_g/corr_samples:0')

## Import the samples that were generated:
samples = np.load(basefolder+modelfolder+'samples.npy')
# plt.plot(samples[:,0],samples[:,1],'--o')
# plt.savefig('latent_samples_big')
# plt.close()

gen_images = sess.run(sampled_images,feed_dict={hidden_activations:samples})
for i in range(50):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(gen_images[i,:,:,:])
    ax[1].set_xlim((0,20))
    ax[1].set_ylim((-20,20))
    ax[1].plot(samples[:i+1,0],samples[:i+1,1],'--o')

    plt.savefig('saveframe'+str(i)+'.png')
    plt.close()
subprocess.call(['ffmpeg', '-framerate', str(30), '-i', 'saveframe%01d.png', '-r', '30','Reconstruct'+'.mp4'])
