# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:38:02 2018

@author: EE-PeiyuanWu
"""
import skvideo.io
import skimage.transform
import csv
import collections
import os 
import numpy as np 
import pandas as pd 
import sys
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.contrib.layers as ly 
import time
import random
from sklearn.manifold import TSNE

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
            frames.append(frame)
        else:
            continue

    return np.array(frames).astype(np.uint8)


def read_video(v_file_path,v_csv_path):
    validation_csv = pd.read_csv(v_csv_path)
    validation_categorical = list(validation_csv[['Video_category']].values.reshape(-1))
    validation_name = list(validation_csv[['Video_name']].values.reshape(-1))
    #validation_label = list(validation_csv[['Action_labels']].values.reshape(-1))

    validation_list = []
    for i in range(len(validation_name)):
        vv = readShortVideo(v_file_path,validation_categorical[i],validation_name[i],downsample_factor=12)
        validation_list.append(vv)    

    return validation_list #, validation_label


validation_list ,  validation_label = read_video(sys.argv[1],sys.argv[2])

parameters = [] 



VGG_MEAN = [103.939, 116.779, 123.68]




def maxpool(name,input_data,trainable=False):
    out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding='SAME',name=name)
    return out 

def conv(name,input_data,out_channel,parameters,trainable=None):
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',[3,3,in_channel,out_channel],dtype=tf.float32,trainable=trainable)
        biases = tf.get_variable('bias',[out_channel],dtype = tf.float32,trainable=trainable)
        conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding='SAME')
        res = tf.nn.bias_add(conv_res,biases)
        out = tf.nn.relu(res,name=name)
        parameters += [kernel,biases]
    return out , parameters



def fc_layer(bottom, name,parameters ,trainable=False):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        weights = tf.get_variable('weights',[x.get_shape()[-1],4096],dtype=tf.float32,trainable=trainable)
        biases = tf.get_variable('bias',[4096],dtype = tf.float32,trainable=trainable)
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        #fc = tf.nn.relu(fc)
        parameters += [weights,biases]

        return fc ,parameters


VGG_MEAN = [103.939, 116.779, 123.68]

tf.reset_default_graph()
img = tf.placeholder(tf.float32,shape=[240,320,3],name='input_image')
lab = tf.placeholder(tf.int64,shape=None)


cc = tf.image.resize_images(img,size=(224,224))
processed_images  = tf.expand_dims(cc, 0)

keep_prob = tf.placeholder(tf.float32)


parameters = []
r , g , b = tf.split(processed_images,3,3)

bgr = tf.concat([b-VGG_MEAN[0],g-VGG_MEAN[1],r-VGG_MEAN[2]],3)


conv1_1 ,parameters= conv('conv1re_1',bgr,64,parameters,trainable=False)
conv1_2 , parameters = conv('conv1_2',conv1_1,64,parameters,trainable=False)
pool1 = maxpool('poolre1',conv1_2,trainable=False)


conv2_1 , parameters = conv('conv2_1',pool1,128, parameters,trainable=False)
conv2_2 , parameters = conv('conwe2_2',conv2_1,128, parameters,trainable=False)
pool2 = maxpool('pool2',conv2_2,trainable=False)


conv3_1 , parameters = conv('conv3_1',pool2,256,parameters,trainable=False)
conv3_2 , parameters = conv('convrwe3_2',conv3_1,256,parameters,trainable=False)
conv3_3 , parameters = conv('convrwe3_3',conv3_2,256,parameters,trainable=False)
pool3 = maxpool('poolre3',conv3_3,trainable=False)


conv4_1 , parameters  = conv('conv4_1',pool3,512,parameters,trainable=False)
conv4_2 , parameters = conv('convrwe4_2',conv4_1,512,parameters,trainable=False)
conv4_3 , parameters = conv('convrwe4_3',conv4_2,512,parameters,trainable=False)
pool4 = maxpool('pool4',conv4_3,trainable=False)


conv5_1 , parameters  = conv('conv5_1',pool4,512,parameters,trainable=False)
conv5_2 , parameters  = conv('convrwe5_2',conv5_1,512,parameters,trainable=False)
conv5_3 , parameters  = conv('convrwe5_3',conv5_2,512,parameters,trainable=False)
pool5 = maxpool('pool5',conv5_3,trainable=False)

fc_6 , parameters = fc_layer(pool5,'fc1',parameters,trainable=False)


def load_weights(weight_file, sess,parameters):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if i < 28:
            print(i, k, np.shape(weights[k]))
            sess.run(parameters[i].assign(weights[k]))

weight_file = 'vgg16_weights.npz'


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
load_weights(weight_file,sess,parameters)





def zero_padding(): 
    emb_1 = []
    for i in validation_list :
        temp = []
        for j in i : 
            q =sess.run(fc_6,feed_dict = {img:j})
            temp.append(q)
        emb_1.append(np.array(temp))
        
    
    emb_v = [] 
    for i in emb_1:
        temp = i.shape
        if temp[0] < 40:
            temp_1 = np.zeros((40-temp[0],4096))
            emb_v.append(np.concatenate((i.reshape(temp[0],4096),temp_1),axis=0))
        else : 
            emb_v.append(i[0:40].reshape(40,4096))
            
    return emb_v


emb_v = zero_padding()



tf.reset_default_graph()


embbed = tf.placeholder(tf.float32,shape=[None,40,4096])
lab = tf.placeholder(tf.int64,shape=[None])
keep_prob = tf.placeholder(tf.float32)


def lstm_cell(units,keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(units, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

cell_bw =  tf.contrib.rnn.MultiRNNCell([lstm_cell(256,keep_prob) for _ in range(4)])
initial_state = cell_bw.zero_state(32,tf.float32)
value , encoder_state = tf.nn.dynamic_rnn(cell_bw, inputs=embbed, initial_state=initial_state, time_major=False)
 

value = tf.transpose(value, [1, 0, 2])  

last = tf.gather(value, int(value.get_shape()[0]) - 1) 


logit = ly.fully_connected(last,128,activation_fn=tf.nn.relu)#,normalizer_fn=ly.batch_norm)


logit_1 = ly.fully_connected(logit,64,activation_fn=tf.nn.relu)#,normalizer_fn=ly.batch_norm)

logit_1 = ly.fully_connected(logit_1,11,activation_fn = None)

prob = tf.nn.softmax(logit_1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_1,labels=tf.one_hot(lab,11)))

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob,1),lab),tf.float32))

#### avoid gradient explorsion !!!!!  
params = tf.trainable_variables()
gradients = tf.gradients(loss,params)
clipped_gradient ,_ = tf.clip_by_global_norm(gradients,5)
optimizer = tf.train.AdamOptimizer(0.0001)
globel_step = tf.Variable(0,name='globel_step',trainable=False)
train_op = optimizer.apply_gradients(zip(clipped_gradient,params),global_step=globel_step)



def next_batch(input_image,label , batch_size=32):
    le = len(input_image)
    #c = np.arange([i for i in range(le)])
    epo = le//batch_size
    temp_5 = le - (batch_size*epo)
    
    for i in range(0,le,32):
        if i == (epo *batch_size) :
            yield np.array(input_image[i:]+input_image[:(32-temp_5)]) , np.array(label[i:]+label[:(32-temp_5)])
        else :
            #yield np.array(input_image[i:i+32]) , np.array(label[i:i+32])
            yield np.array(input_image[i:i+32]) , np.array(label[i:i+32])

sess= tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)


def validation_test():
    for i in range(1):
        acc_trace=[]
        for m,n in next_batch(emb_v,validation_label):
            acc_1= sess.run(acc,feed_dict={embbed:m,lab:n,keep_prob:1.0})
            acc_trace.append(acc_1)
    print(np.mean(acc_trace))
    


saver.restore(sess,'5')
batch_size = 32
fans=[]
count= 0
le = len(emb_v)//batch_size
ty = len(emb_v) - (batch_size *le )
for m,n in next_batch(emb_v,validation_label):
    if count < le : 
        fans+=list(np.argmax(sess.run(prob,feed_dict={embbed:m,lab:n,keep_prob:1}),1))
        count+=1
    else : 
        
        fans+=list(np.argmax(sess.run(prob,feed_dict={embbed:m,lab:n,keep_prob:1}),1))[0:ty]

pp = open(os.path.join(sys.argv[3],'p2_result.txt'),'w')
for i in fans: 
    pp.write(str(i)+'\n')

'''
feature_vector = []
for m,n in next_batch(emb_v,validation_label):
    feature_vector.append(sess.run(logit,feed_dict={embbed:m,lab:n,keep_prob:1}))
    
count= 0 
for i in feature_vector : 
    if count ==0 : 
        f_ma = i 
        count+=1
        continue
    f_ma = np.concatenate((f_ma,i),axis=0)




X_embedded = TSNE(n_components=2,random_state=9).fit_transform(f_ma)


color = ['b','g','r','c','m','y','k','silver']
shape = ["C0",'C2','C5']
for i , j ,k  in zip(X_embedded[:,0],X_embedded[:,1],validation_label):
    if k <= 7 : 
        plt.scatter(i,j,s=1.5,color = color[k])
    else : 
        #plt.scatter(i,j,s=1.5,color=color[10-k],marker=shape[10-k])
        plt.scatter(i,j,s=1.5,color=shape[10-k])    
plt.savefig(os.path.join('ans','RNN_feature.png'))
'''