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
modelfolder = 'Video_ball_color_statvar_ckpts/'
data = 'modelep999.ckpt'
metadata = data+'.meta'
saver = tf.train.import_meta_graph(basefolder+modelfolder+metadata)
saver.restore(sess,basefolder+modelfolder+data)

## Name vars to restore:
graph = tf.get_default_graph()
sampled_images = graph.get_tensor_by_name("model_g/deconv2d_4/Sigmoid:0")
hidden_activations = graph.get_tensor_by_name('model_g/corr_samples_aug:0')
#
# ## For the case of the static variable:
# mean_weights = graph.get_tensor_by_name('model_g/Stat_weights_mean:0')
# var_weights = graph.get_tensor_by_name('model_g/Stat_weights_var:0')

# params = sess.run([mean_weights,var_weights])
# print(params[0].shape)
# plt.plot(params[0][0,:])
# plt.savefig('means.png')
# plt.close()
# plt.plot(params[1][0,:])
# plt.savefig('vars.png')
# plt.close()

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
# ax[0].axvline(x = 81)
# ax[1].axvline(x = 81)
# ax[0].axvline(x = 296)
# ax[1].axvline(x = 296)
# plt.savefig('z_evolution.png')
# plt.close()



## Import the samples that were generated:
samples = np.load(basefolder+modelfolder+'samples.npy')

# plt.plot(samples[:,0],samples[:,1],'--o')
# plt.savefig('latent_samples_big')
# plt.close()

# We replace the dynamical samples with their means:
samples[:,0:2] = np.tile(np.mean(samples[:,0:2],axis = 0),(50,1))

xmin = min(samples[:,0])
xmax = max(samples[:,0])
ymin = min(samples[:,1])
ymax = max(samples[:,1])

gen_images = sess.run(sampled_images,feed_dict={hidden_activations:samples})
for i in range(50):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(gen_images[i,:,:,:])
    ax[1].set_xlim((xmin-2,xmax+2))
    ax[1].set_ylim((ymin-2,ymax+2))
    ax[1].plot(samples[:i+1,0],samples[:i+1,1],'--o')

    plt.savefig('saveframe'+str(i)+'.png')
    plt.close()
subprocess.call(['ffmpeg', '-framerate', str(10), '-i', 'saveframe%01d.png', '-r', '30','Reconstruct_'+modelfolder.split('/')[0]+'.mp4'])
