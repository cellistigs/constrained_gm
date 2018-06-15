## A place to store all the costs that we implement:
import tensorflow as tf
import numpy as np


## Reconstruction Cost:
def likelihood_cost(true_images,gen_images):
    ## Let's impose a gaussian likelihood elementwise and call the log likelihood
    ## an RMSE:
    resvec_x = true_images-gen_images
    diff = tf.reduce_mean(tf.matmul(tf.transpose(true_images-gen_images),true_images-gen_images))
