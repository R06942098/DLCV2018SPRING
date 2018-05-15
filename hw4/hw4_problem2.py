# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:37:47 2018

@author: bowei
"""
import numpy as np
import tensorflow as tf
import os 
import tensorflow.contrib.layers as ly
import matplotlib.pyplot as plt 
import time
from skimage import io 
import random
from scipy.misc import imread, imresize
from sklearn.decomposition import PCA
from PIL import Image, ImageOps
import pandas as pd 
import pickle 
import sys

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    

def plot(x):
    x = x - np.min(x)
    x /= np.max(x)
    x *= 255 
    x= x.astype(np.uint8)
    x = x.reshape(64,64,3)
    return x 

batch_size = 64

tf.reset_default_graph()

real_image = tf.placeholder(tf.float32,shape=(None,64,64,3))

noise = tf.placeholder(tf.float32,shape=(None,100))

channel = 3 
def generator_conv(z):
    train = ly.fully_connected(
        z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4,512 ))
    train = ly.conv2d_transpose(train, 256, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 32, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, channel, 5, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    print(train.name)
    return train

def dis_conv(img, reuse=False):
    with tf.variable_scope('dis_conv') as scope:
        if reuse:
            scope.reuse_variables()
        size = 32
        img = ly.conv2d(img, num_outputs=size, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        source_logit = ly.fully_connected(tf.reshape(
            img, [batch_size, 4*4*256]), 1,activation_fn=None)

    return source_logit

with tf.variable_scope('generator_conv'):
    sythetic_image = generator_conv(noise)


tf.summary.image('sythetic_image',sythetic_image)

logits_fake = dis_conv(sythetic_image,reuse=False)

logits_real = dis_conv(real_image,reuse=True)


fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),logits=logits_fake))


real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),logits=logits_real))

d_loss = fake_loss + real_loss

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),logits=logits_fake))


theta_g = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_conv')
    
theta_c = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis_conv')



counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

g_opt = tf.train.AdamOptimizer(0.001).minimize(g_loss,var_list=theta_g)

counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

d_opt = tf.train.AdamOptimizer(0.001).minimize(d_loss,var_list=theta_c)




def next_batch(input_image , batch_size=64):
    le = len(input_image)
    np.random.shuffle(input_image)
    epo = le//batch_size
    for i in range(0,epo*batch_size,64):
        yield np.array(input_image[i:i+64])



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

saver.restore(sess,'140epochs')

with open('dc_no_major.pickle', 'rb') as handle:
    no = pickle.load(handle)
    
#no = np.random.normal(size=(32,100))
rs= sess.run(sythetic_image,feed_dict={noise:no})
overall = []
for i in range(4):
    temp = []
    for j in range(8):
        temp.append(plot(rs[i * 8 + j]))
    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)
res = np.squeeze(res)
plt.imsave(os.path.join(sys.argv[2],'fig2_3.jpg'),res)
#plt.figure(figsize=[8, 4])
#plt.imshow(res)
#plt.show()   


plt.figure()
plt.subplot(2,1,1)
temp = pd.read_csv('dc_gl.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('loss of gnerator')
plt.plot(temp)
#plt.savefig(os.path.join(sys.argv[2],'fig2_1.jpg')

plt.subplot(2,1,2)
temp = pd.read_csv('dc_dl.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('loss of discriminator')
plt.plot(temp)
plt.savefig(os.path.join(sys.argv[2],'fig2_2.jpg'))