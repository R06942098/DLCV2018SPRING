# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:46:31 2018

@author: bowei
"""

import tensorflow as tf 
import tensorflow.contrib.layers as ly 
import matplotlib.pyplot as plt 
from skimage import io
import numpy as np 
from sklearn.decomposition import PCA
import os 
from scipy.misc import imread, imresize
from PIL import Image, ImageOps
import os 
import random 
import time 
from sklearn.metrics import mean_squared_error
import pandas as pd 
from sklearn.manifold import TSNE
import pickle
import sys

##'hw4_data'

test_file_path = 'hw4_data/test'


def load_testdata(file_path):
    temp = os.listdir(file_path)
    image_list = []
    for i in temp : 
        kk = os.path.join(file_path,i)
        te = imread(kk)
        te = (te/127.5)-1
        image_list.append(te)
    return image_list

test_image_list = load_testdata(os.path.join(sys.argv[1],os.listdir(sys.argv[1])[0]))

label = pd.read_csv(os.path.join(sys.argv[1],os.listdir(sys.argv[1])[1]))
label = label[['Male']].values

#te_image = test_image_list[15000:]


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def plot(x):
    x = x.flatten()
    x = x - np.min(x)
    x /= np.max(x)
    x *= 255 
    x= x.astype(np.uint8)
    x = x.reshape(64,64,3)
    return x 


def next_batch(input_image , batch_size=64):
    le = len(input_image)
    epo = le//batch_size
    for i in range(0,batch_size*epo,64):
        yield np.array(input_image[i:i+64])
        
tf.reset_default_graph()



real_data = tf.placeholder(tf.float32,shape=(None,64,64,3))

def encoder(real_data,activation):
    
    conv1 = ly.conv2d(real_data,64,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

    conv2 = ly.conv2d(conv1,128,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

    conv3 = ly.conv2d(conv2,256,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

    conv4 = ly.conv2d(conv3,512,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
    return conv4

##64 6 8 8 
def decoder(x,activation):
    x = ly.fully_connected(x ,64*4*8*8, activation_fn=tf.nn.relu,normalizer_fn = ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))
    x = tf.reshape(x,shape=[-1,8,8,64*4])
    #unsample1 = ly.conv2d_transpose(x,512,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

    upsample2 = ly.conv2d_transpose(x,256,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

#unsample3 = ly.conv2d_transpose(upsample2,1,kernel_size=4,stride=1,padding='SAME',activation_fn=lrelu,normalizer_fn=ly.batch_norm)
    upsample4 = ly.conv2d_transpose(upsample2,128,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))


    upsample5 = ly.conv2d_transpose(upsample4 ,32,kernel_size=5,stride=2,padding='SAME',activation_fn=activation,normalizer_fn=ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

    upsample6 = ly.conv2d_transpose(upsample5 ,3,kernel_size=5,stride=1,padding='SAME',activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.02))
    
    return upsample6

conv4 = encoder(real_data , tf.nn.leaky_relu)

flat = tf.contrib.layers.flatten(conv4)

mu = ly.fully_connected(flat,128, activation_fn=tf.identity , normalizer_fn = ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02))

epsilon = tf.random_normal(shape=(64,128), mean=0.0, stddev=1.0)

steddv = ly.fully_connected(flat,128, activation_fn=tf.nn.softplus , normalizer_fn = ly.batch_norm,weights_initializer=tf.random_normal_initializer(0, 0.02)) + 1e-6

z  = tf.add(mu,tf.multiply(tf.sqrt(tf.exp(steddv)),epsilon))

upsample6 = decoder(z,tf.nn.leaky_relu)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

saver = tf.train.Saver()

saver.restore(sess,'vae')

pp = test_image_list[0:64]
feed_dict = {real_data:pp,epsilon:np.zeros((64,128))}
reco = sess.run(upsample6,feed_dict=feed_dict)
rr = reco[0:10]
pp = test_image_list[0:10]
pp = np.array(pp)
tt = np.concatenate((pp,rr),axis=0)

overall = [] 
for i in range(2):
    temp = []
    for j in range(10):
        temp.append(plot(tt[i * 10 + j]))
    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)
res = np.squeeze(res)
plt.imsave(os.path.join(sys.argv[2],'fig1_3.jpg'),res)
#plt.figure(figsize=[10, 2])
#plt.imshow(res)
#plt.show()



##讀取 noise 
with open('vae_no.pickle', 'rb') as handle:
    no = pickle.load(handle)

#no = np.random.normal(size=(64,128))
feed_dict = {z:no}
reco = sess.run(upsample6,feed_dict=feed_dict)
reco = reco[0:32]
overall = [] 
for i in range(4):
    temp = []
    for j in range(8):
        temp.append(plot(reco[i * 4 + j]))
    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)
res = np.squeeze(res)
plt.imsave(os.path.join(sys.argv[2],'fig1_4.jpg'),res)
#res = (res+1)/2
#plt.figure(figsize=[8, 4])
#plt.imshow(res)
#plt.show()

plt.figure()
plt.subplot(2,1,1)
temp = pd.read_csv('vae_kl.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('KLD')
plt.plot(temp)
#plt.savefig('fig1_1.jpg')

plt.subplot(2,1,2)
temp = pd.read_csv('vae_mse.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('Mean square error')
plt.plot(temp)
plt.savefig(os.path.join(sys.argv[2],'fig1_2.jpg'))




p_4 = np.array(test_image_list)
lat = []
for i in next_batch(test_image_list):
    lat.append(sess.run(z,feed_dict={real_data:i,epsilon:np.zeros((64,128))}))
count = 0
for i in lat : 
    if count == 0:
        ff = i
        count+=1
        continue
    ff = np.concatenate((ff,i),axis=0)

label = label[:len(ff)]
    
X_embedded = TSNE(n_components=2,random_state=9).fit_transform(ff)

plt.figure()
for i , j  in zip(X_embedded,label):
    if j == 1 : 
        plt.scatter(i[0],i[1],s=1.2,color='b')
    else : 
        plt.scatter(i[0],i[1],s=0.8,color='r')
plt.title('Gender')
plt.savefig(os.path.join(sys.argv[2],'fig1_5.jpg'))


    

