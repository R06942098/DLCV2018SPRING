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
import sys
import pickle

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


tf.reset_default_graph()


def plot(x):
    x = x - np.min(x)
    x /= np.max(x)
    x *= 255 
    x= x.astype(np.uint8)
    x = x.reshape(64,64,3)
    return x    
batch_size =  64

channel = 3 
def generator_conv(z,labels):

    labels_one_hot = tf.one_hot(labels, 2)  ##focus on the label we have !!

    z_labels = tf.concat([z, labels_one_hot],1)

    train = ly.fully_connected(
       z_labels, 4 * 4 * 1024, activation_fn=lrelu,normalizer_fn=ly.batch_norm)
    #train = tf.layers.Dense(z_labels,8*8*256,activation=lrelu)

    train = tf.reshape(train, (-1, 4, 4, 1024))
    ##多加一層 並且改 lrelu
    train = ly.conv2d_transpose(train, 512, 5, stride=2,
                                activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

    train = ly.conv2d_transpose(train, 256, 5, stride=2,
                                activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 5, stride=2,
                                activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 32, 5, stride=2,
                                activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, channel, 5, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    print(train.name)
    return train

def dis_conv(img, reuse=False):
    with tf.variable_scope('dis_conv') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        img = ly.conv2d(img, num_outputs=size, kernel_size=5,padding='SAME',normalizer_fn=ly.batch_norm,
                        stride=2, activation_fn=tf.nn.leaky_relu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
        source_logit = ly.fully_connected(tf.reshape(
            img, [batch_size, 4*4*512]), 1, activation_fn=None)
        class_logit = ly.fully_connected(tf.reshape(
            img, [batch_size, 4*4*512]),2,activation_fn=None,scope='q_H')

    return source_logit , class_logit 


noise = tf.placeholder(tf.float32,shape=(None,100))

real_image = tf.placeholder(tf.float32,shape=(None,64,64,3))

label = tf.placeholder(tf.int32,shape=[batch_size])

            
with tf.variable_scope('generator_conv'):
    sythetic_image  = generator_conv(noise,label)


logits_fake , label_fake = dis_conv(sythetic_image,reuse=False)

logits_real , label_real = dis_conv(real_image,reuse=True)

fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.zeros_like(logits_fake)))

real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels=tf.ones_like(logits_real)))


class_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=label_real,
                                                labels=tf.one_hot(label,2)))

class_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=label_fake,
                                                labels=tf.one_hot(label,2)))

d_loss = fake_loss + real_loss + class_loss_fake + class_loss_real

fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.ones_like(logits_fake)))

g_loss = fake + class_loss_fake + class_loss_real

def next_batch(input_image,labels , batch_size=64):
    le = len(input_image)
    epo = le//batch_size
    for i in range(0,epo*batch_size,64):
        #yield temp_1[i:i+64] ,temp_2[i:i+64]
        yield np.array(input_image[i:i+64]) ,labels[i:i+64]


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess,'info_ma')

with open('info_long_short.pickle', 'rb') as handle:
    no = pickle.load(handle)


label_t = np.array([0 for i in range(10)]+[1 for i in range(10)]+[0 for i in range(44)])
#label_p = np.array([0 for i in range(10)]+[1 for i in range(10)]+[0 for i in range(108)])
ts = sess.run(sythetic_image,feed_dict={noise:no,label:label_t})
rs = ts[:20,:,:,:]

overall = []
for i in range(2):
    temp = []
    for j in range(10):
        temp.append(plot(rs[i * 10 + j]))

    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)

res = np.squeeze(res)
plt.imsave(os.path.join(sys.argv[2],'fig4_3.jpg'),res)
#res = (res+1)/2
#plt.figure(figsize=[8, 2])
#plt.imshow(res)
#plt.show()


plt.figure()
plt.subplot(3,1,1)
temp = pd.read_csv('info_conditional_entropy.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('conditional entropy')
plt.plot(temp)
#plt.savefig('fig1_1.jpg')

plt.subplot(3,1,2)
temp = pd.read_csv('info_d_loss.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('d_loss')
plt.plot(temp)

plt.subplot(3,1,3)
temp = pd.read_csv('info_g_loss.csv')[['Value']]
plt.xlabel('epochs')
plt.ylabel('g_loss')
plt.plot(temp)

plt.savefig(os.path.join(sys.argv[2],'fig4_2.jpg'))