# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:29:24 2018

@author: EE-PeiyuanWu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:09:39 2018

@author: EE-PeiyuanWu
"""


#from HW5_data import reader 
import os 
import numpy as np 
import pandas as pd 
import sys
import matplotlib.pyplot as plt 
import tensorflow as tf 
#from slim.nets import inception_resnet_v2 
#import inception_preprocessing
import tensorflow.contrib.layers as ly 
import time
import random
from tensorflow.python.layers import core as layer_core
from scipy.misc import imread, imresize , imrotate
from PIL import Image, ImageOps
#import cv2


slim = tf.contrib.slim


### padding is not the good way to fullfill this task. 

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]='1'
####  40 is the best , training epochs need to be about 100~150epochs 
## maxlen  : 2995 ( include the BOS )


#v_name =os.listdir(sys.argv[1])
v_name = os.listdir(sys.argv[1])
'''
def preprocess_data():
    t_label_path = 'HW5_data/FullLengthVideos/labels/train'
    v_label_path = 'HW5_data/FullLengthVideos/labels/valid'

    temp_1 = os.listdir(t_label_path)

    t_label_list = [] 
    for i in temp_1 : 
        gg = [] 
        ff = open(os.path.join(t_label_path,i))
        for j in ff : 
            gg.append(int(j[:-1]))
        t_label_list.append(gg)

    temp_1 = os.listdir(v_label_path)
    v_label_list = [] 
    for i in temp_1 : 
        gg = [] 
        ff = open(os.path.join(v_label_path,i))
        for j in ff : 
            gg.append(int(j[:-1]))
        v_label_list.append(gg)
    
    file_path_1 = 'HW5_data/FullLengthVideos/videos/train'
    file_path_2 = 'HW5_data/FullLengthVideos/videos/valid'
    
    temp_1 = os.listdir(file_path_1)
    
    t_image_list = []
    for i in temp_1 : 
        temp_2 = os.path.join(file_path_1,i)
        temp_3 = os.listdir(temp_2)
        gg = []
        for j in temp_3 :
            gg.append(imread(os.path.join(temp_2,j)))
        t_image_list.append(gg)
    
    v_image_list = []
    
    temp_1 = os.listdir(file_path_2)
    for i in temp_1 : 
        temp_2 = os.path.join(file_path_2,i)
        temp_3 = os.listdir(temp_2)
        gg = []
        for j in temp_3 : 
            gg.append(imread(os.path.join(temp_2,j)))
        v_image_list.append(gg)
    
        
    return t_image_list , v_image_list , t_label_list , v_label_list 
'''

            
#t_image_list , v_image_list , t_label_list , v_label_list= preprocess_data()

def preprocess_data(file_path):    
    temp_1 = os.listdir(file_path)
    
    t_image_list = []
    for i in temp_1 : 
        temp_2 = os.path.join(file_path,i)
        temp_3 = os.listdir(temp_2)
        gg = []
        for j in temp_3 :
            gg.append(imread(os.path.join(temp_2,j)))
        t_image_list.append(gg)
    
    return t_image_list
v_image_list = preprocess_data(sys.argv[1])


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
#input_length = tf.placeholder(tf.int32,shape=None)
img = tf.placeholder(tf.float32,shape=[240,320,3],name='input_image')
lab = tf.placeholder(tf.int64,shape=None)
#saver.restore(sess,'inception_resnet_v2_2016_08_30.ckpt')


cc = tf.image.resize_images(img,size=(224,224))
processed_images  = tf.expand_dims(cc, 0)

keep_prob = tf.placeholder(tf.float32)


parameters = []
#lab = tf.placeholder(tf.int32,shape=[None,512,512,1],name='annotation')
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

def not_padding():
    emb_1 = []
    for i in v_image_list :
        temp = []
        for j in i : 
            q =sess.run(fc_6,feed_dict = {img:j})
            temp.append(q.reshape(1,-1))
        emb_1.append(np.array(temp))
    
    return emb_1 


#emb_t , emb_v = zero_padding()


emb_v = not_padding()

def sample(emb_t,t_label_list,sample_size=350):
    #temp = [i for i in range(len(emb_t))]
    le = len(emb_t) - 175
    temp = [i for i in range(le) if i >=175]
    #temp_2 = random.sample(temp,1)[0]
    #train_t = emb_t[np.array(temp_2),:]
    
    
    for temp_2 in temp : 
        train_t = emb_t[(temp_2-175):(temp_2+175),:]
    
        t_label_list = [11] + t_label_list + [12]
        temp_3 = np.array(t_label_list).reshape(-1,1)
        temp_4 = temp_3[(temp_2-175):(temp_2+175),:]
        temp_5 = temp_3[(temp_2-174):(temp_2+176),:] 
        #decoder_input = [11]
        #decoder_output = []
        decoder_input = []
        decoder_output = []    
        for i in temp_4: 
            decoder_input.append(i[0])
        for i in temp_5 :
            decoder_output.append(i[0])
        
        #decoder_output.append(12)
        de_in = np.array(decoder_input).reshape(1,350)
        de_out = np.array(decoder_output).reshape(1,350)
    
        yield train_t.reshape(1,sample_size,4096) , de_in , de_out
    
    
    



tf.reset_default_graph()


batch_size = 1



#### 2994 實在是太長了，看起來是完全訓練不起來!!!  

#### TA advised us to random sample 250~500 steps to train this model .


#embbed = tf.placeholder(tf.float32,shape=[None,350,4096])
#decoder_length = tf.placeholder(tf.int32,shape=[None])
#decoder_inputs = tf.placeholder(tf.int64,shape=[None,351])
#decoder_outputs = tf.placeholder(tf.int64,shape=[None,351])
#keep_prob = tf.placeholder(tf.float32)



def lstm_cell(units,keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(units, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)



#### BOS : 11 , EOS:12 PAD:13
embeddings_matrix = np.zeros((13,13))
for i in range(13):
    embeddings_matrix[i][i] =1
    



def construct_graph(mode,embedding_matrix):
    dim = 256
    batch_size = 1
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    embbed = tf.placeholder(tf.float32,shape=[1,350,4096])
    
    decoder_length = tf.placeholder(tf.int32,shape=[1])
    decoder_inputs = tf.placeholder(tf.int64,shape=[1,350])
    decoder_outputs = tf.placeholder(tf.int64,shape=[1,350])
    keep_prob = tf.placeholder(tf.float32)


    Inp = (embbed,decoder_inputs,decoder_length,decoder_outputs,keep_prob)
    emb_x_y = tf.nn.embedding_lookup( embedding_matrix , decoder_inputs )
    emb_x_y = tf.cast(emb_x_y,tf.float32)

    with tf.name_scope("Encoder"):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(256,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(256,reuse=tf.get_variable_scope().reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

        enc_rnn_out , enc_rnn_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , embbed ,dtype=tf.float32)#,swap_memory=True)
        enc_rnn_out = tf.concat(enc_rnn_out, 2)

        c = tf.concat([enc_rnn_state[0][0],enc_rnn_state[1][0]],axis=1)
        h = tf.concat([enc_rnn_state[0][1],enc_rnn_state[1][1]],axis=1)

        enc_rnn_state = tf.contrib.rnn.LSTMStateTuple(c,h)



## less the time_major !!!!

    with tf.variable_scope("Decoder") as decoder_scope:
        projection_layer = layer_core.Dense(13 ,use_bias=False)
        mem_units = 2*dim
        #out_layer = Dense( 2449)
        #batch_size = tf.shape(enc_rnn_out)[0]
        beam_width = 3

    
        num_units = 2*dim
        memory = enc_rnn_out

        if mode == "infer":

            memory = tf.contrib.seq2seq.tile_batch( memory, multiplier=beam_width )
            decoder_length = tf.contrib.seq2seq.tile_batch( decoder_length, multiplier=beam_width)
            enc_rnn_state = tf.contrib.seq2seq.tile_batch( enc_rnn_state, multiplier=beam_width )
            batch_size = batch_size * beam_width

        else:
            batch_size = batch_size

        #attention_states = tf.transpose(enc_rnn_out,[1,0,2])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention( num_units,memory,memory_sequence_length=decoder_length)#,normalize=True)

        cell = tf.contrib.rnn.BasicLSTMCell( 2*dim )
        cell= tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)        
        #lstm = tf.contrib.rnn.BasicLSTMCell(256, reuse=tf.get_variable_scope().reuse)
        #lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        #cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(2)])

        cell = tf.contrib.seq2seq.AttentionWrapper( cell,
                                                attention_mechanism,
                                                attention_layer_size=num_units,
                                                name="attention")

        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone( cell_state=enc_rnn_state)



        if mode == "train":

            helper = tf.contrib.seq2seq.TrainingHelper( inputs = emb_x_y , sequence_length = decoder_length )
            decoder = tf.contrib.seq2seq.BasicDecoder( cell = cell, helper = helper, initial_state = decoder_initial_state,output_layer=projection_layer) 
            outputs, final_state, final_sequence_lengths= tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                           scope=decoder_scope)

            logits = outputs.rnn_output
            sample_ids = outputs.sample_id

        else:
            
            emb = tf.cast(embedding_matrix,tf.float32)
            #de_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state,multiplier=9) 
            #start_tokens = tf.tile(tf.constant([dictionary['BOS']], dtype=tf.int32), [ batch_size ] )
            #end_token = 0

            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = cell,
                                                               embedding = emb,
                                                               start_tokens = tf.fill([1],11) ,
                                                               end_token = 12,
                                                               initial_state = decoder_initial_state,
                                                               beam_width = beam_width,
                                                               output_layer = projection_layer )

            outputs, t1 , t2 = tf.contrib.seq2seq.dynamic_decode(  my_decoder,
                                                                   maximum_iterations=350,scope=decoder_scope )

            logits = tf.no_op()
            #sample_ids = outputs.rnn_outputs
            sample_ids = outputs.predicted_ids

    
    if mode == "train":
        #non_zero = tf.cast(tf.not_equal(target_label,0),tf.float32)
        
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_label,logits=logits)
        la = tf.nn.embedding_lookup( embedding_matrix , decoder_outputs )
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=la,logits=logits))
        #loss = tf.reduce_sum(loss*non_zero)/tf.reduce_sum(non_zero)

        globel_step = tf.Variable(0,name='globel_step',trainable=False)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss,params)
        clipped_gradient ,_ = tf.clip_by_global_norm(gradients,5)
        #optimizer = tf.train.AdamOptimizer(0.0001)

        optimizer = tf.train.GradientDescentOptimizer(0.001)
        ##clipped gradients is this way to use !!!!!
        train_op = optimizer.apply_gradients(zip(clipped_gradient,params),global_step=globel_step)
        #train_op = optimizer.minimize(loss)
        #or sparse soft_max
        #output_vocab_size = len(dictionary)

        #loss = tf.losses.softmax_cross_entropy(  tf.one_hot( target_label,output_vocab_size ) , logits )
        #train_op = tf.train.AdamOptimizer().minimize(loss)
        ### clip gradient 
        sample_ids = tf.cast(sample_ids,tf.int64)
        correct = tf.reduce_sum( tf.cast( tf.equal( sample_ids ,decoder_outputs ) , dtype=tf.float32 ) ) / 350
         #sample_ids = tf.transpose( sample_ids , [2,0,1] )[0]
    else : 
        correct = None
         #correct = tf.reduce_sum( tf.cast( tf.equal( sample_ids , Y ) , dtype=tf.float32 ) ) / maxlen
        loss = None
        train_op = None
        
    return train_op , loss , correct , sample_ids , logits , Inp 

'''
tensorboard_dir = 'seq2seq/'   
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


writer = tf.summary.FileWriter(tensorboard_dir)     
'''


def next_batch(input_image,label,decoder_output , batch_size=32):
    le = len(input_image)
    #c = np.arange([i for i in range(le)])
    epo = le//batch_size
    #last  = epo*batch_size
    #random.shuffle(input_image)
    #for i in range(0,batch_size*epo,16):
    #    yield np.array(encoder_input[i:i+16])
    c = list(zip(input_image,label,decoder_output))
    random.shuffle(c)
    temp , temp_1 ,temp_2 = zip(*c)
    for i in range(0,epo*batch_size,32):
        #if i == (epo *batch_size) :
        #    yield np.array(input_image[i:]) , np.array(label[i:])
        #else :
        #yield np.array(input_image[i:i+32]) , np.array(label[i:i+32]) , np.array(decoder_output[i:i+32])
        yield np.array(temp[i:i+32]) , np.array(temp_1[i:i+32]) ,  np.array(temp_2[i:i+32])
        

'''

def validation_test():
    acc_trace = []
    for k in range(len(emb_v)): 
        m,n,l = sample(emb_v[k],v_label_list[k])
        feed_dict={Inp[0]:m,Inp[1]:n,Inp[3]:l,Inp[2]:np.ones([1])*350,Inp[4]:1}

        acc_1 = train_sess.run([correct],feed_dict=feed_dict)
        acc_trace.append(acc_1)
    print(np.mean(acc_trace))

tf.reset_default_graph()

train_graph = tf.Graph()
infer_graph = tf.Graph()




with train_graph.as_default():

    train_op, loss , correct,sample_ids,logits ,Inp = construct_graph("train",embeddings_matrix)
    initializer = tf.global_variables_initializer()
    train_saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tensorboard_dir)   

train_sess = tf.Session(graph=train_graph)
writer.add_graph(train_sess.graph)
train_sess.run(initializer)
train_saver.restore(train_sess,'temp/model_1')  ## best model

count = 0
for _ in range(10):
    for i in range(len(emb_t)):
        for m,n,l in sample(emb_t[i],t_label_list[i]):
            feed_dict={Inp[0]:m,Inp[1]:n,Inp[3]:l,Inp[2]:np.ones([1])*350,Inp[4]:0.5}
            acc_1,loss_1,_ = train_sess.run([correct,loss,train_op],feed_dict=feed_dict)
        if count % 3 == 0 : 
            #validation_test()
            print(loss_1)
    count+=1
    print(count)
train_saver.save(train_sess,'temp/model_3')
'''

tf.reset_default_graph()
infer_graph = tf.Graph()

with infer_graph.as_default():

    _ , _ , correct_test , pred_ids,_,Inp = construct_graph("infer",embeddings_matrix)

    infer_saver = tf.train.Saver()

infer_sess = tf.Session(config=tf.ConfigProto(),graph=infer_graph)

model_file= 'model_1'
infer_saver.restore(infer_sess, model_file)

def inference(emb_t,sample_size=350):
    #temp = [i for i in range(len(emb_t))]
    qq= []
    for i in range(0,len(emb_t),sample_size):
        qq.append(emb_t[i:i+350])
    
    if qq[-1].shape[0]<350:
        temp = 350 - qq[-1].shape[0]
        qq[-1] = np.concatenate((qq[-1],np.zeros((temp,4096))))
    return qq , temp

def acc_compute(y_true,y_pred):
    acc = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            acc.append(1)
    print(len(acc)/len(y_true))
    #return acc

for i in range(5):
    ###open file 
    m,temp = inference(emb_v[i].reshape(-1,4096))
    acc_trace = []
    pre_la = []
    #ans = []
    for k in range(1):
        ans = []
        count = 0
        for j in range(len(m)):
            if count == (len(m)-1):
                feed_dict={Inp[0]:m[j].reshape(1,350,4096),Inp[2]:np.ones([1])*350,Inp[4]:1}
                prob_1= infer_sess.run(pred_ids,feed_dict=feed_dict)
                #hh= np.argmax(prob_1[0],1)
                ans += list(prob_1[0][:,k][:350-temp])
                #re = acc_compute(hh,l[j])
                #re = re[:350-temp]
                #temp_2 = []
                #for p in re : 
                #    if p == 1 :
                #        temp_2.append(1)
                #acc_trace.append(len(temp_2)/(350-temp))
                #pre_la.append(hh)
    
            else:
                ## no last output --
                feed_dict={Inp[0]:m[j].reshape(1,350,4096),Inp[2]:np.ones([1])*350,Inp[4]:1}
                prob_1 = infer_sess.run(pred_ids,feed_dict=feed_dict)
                #hh = np.argmax(prob_1[0],1)
                ans += list(prob_1[0][:,k])
     
                #acc_trace.append(acc_1)
                #pre_la.append(hh)
            count+=1
    opp = open(os.path.join(sys.argv[2],v_name[i] + '.txt'),'w')
    for i in ans:
    	opp.write(str(i)+'\n')
    #acc_trace_1.append(np.mean(acc_trace))
    #pre_trace_1.append(pre_la)
    

def acc_compute(y_true,y_pred):
    acc = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            acc.append(1)
    print(len(acc)/len(y_true))
    return acc
#saver.restore(sess,'normal')
'''
laaa = v_label_list[1][:350]
pic =  v_image_list[1][:350]#.reshape(350,-1)
aaa = ans[0:350]
count=0
for i in range(0,350,10):
    plt.imsave('pic_2/'+str(count)+'img.png',res[i*24:i*24+240])
    plt.imsave('pic_2/'+str(count)+'pred.png',np.array(aaa[i:i+10]).reshape(1,-1))
    plt.imsave('pic_2/'+str(count)+'original.png',np.array(laaa[i:i+10]).reshape(1,-1))
    count+=1
'''
############ testing set ####################

'''
def plot(x):
    x = x - np.min(x)
    x /= np.max(x)
    x *= 255 
    x= x.astype(np.uint8)
    x = x.reshape(64,64,3)
    return x    


overall = []
for i in range(35):
    temp = []
    for j in range(10):
        temp.append(pic[i * 10 + j])
    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)
#     res = cv2.cvtColor((res)*255, cv2.COLOR_GRAY2BGR)
#     cv2.imwrite('sample.png', res)
res = np.squeeze(res)

pre_test = []
for i in range(0,len(emb_v),350):
    pre_test.append(emb_v[0].reshape(2140,4096)[i:i+350])
    


acc_trace = []
for _ in range(10):
    m,n,l = sample(emb_v[4],v_label_list[4])
    acc_1= sess.run(acc,feed_dict={embbed:m,\
                          decoder_inputs:n,\
                          decoder_outputs:l,\
                          decoder_length: np.ones((1))*351,keep_prob:1})
    acc_trace.append(acc_1)
    #print(acc_1)


def inference(emb_t,t_label_list,sample_size=350):
    #temp = [i for i in range(len(emb_t))]
    qq= []
    for i in range(0,len(emb_t),sample_size):
        qq.append(emb_t[i:i+350])
    
    if qq[-1].shape[0]<350:
        temp = 350 - qq[-1].shape[0]
        qq[-1] = np.concatenate((qq[-1],np.zeros((temp,4096))))
    
    gg= []
    ff = []
    for i in range(0,len(emb_t),sample_size):
        gg.append(t_label_list[i:i+350])
        ff.append(t_label_list[i:i+350])
    
    for i in range(len(gg)):
        gg[i] = [11]+gg[i]
        ff[i] = ff[i] + [12]
    
    gg[-1] = gg[-1] + [0]*temp
    ff[-1] = ff[-1] + [0]*temp
    #temp_2 = random.sample(temp,sample_size)
    #de_in = np.array(decoder_input).reshape(1,251)
    #de_out = np.array(decoder_output).reshape(1,251)
    
    return qq,gg,ff,temp

def acc_compute(y_true,y_pred):
    acc = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            acc.append(1)
        else : 
            acc.append(0)
    return acc

acc_trace_1 = []
pre_trace_1 = []
for i in range(5):
    ###open file 
    m,n,l,temp = inference(emb_v[i].reshape(-1,4096),v_label_list[i])
    acc_trace = []
    pre_la = []
    count = 0
    ans = []
    for j in range(len(m)):
        if count == (len(m)-1):
            prob_1= sess.run(prob,feed_dict={embbed:m[j].reshape(1,350,4096),\
                          decoder_inputs:np.array(n[j]).reshape(1,351),\
                          decoder_outputs:np.array(l[j]).reshape(1,351),\
                          decoder_length: np.ones((1))*351,keep_prob:1})
            hh= np.argmax(prob_1[0],1)
            ans += list(hh[:350-temp])
            re = acc_compute(hh,l[j])
            re = re[:350-temp]
            temp_2 = []
            for p in re : 
                if p == 1 :
                    temp_2.append(1)
            acc_trace.append(len(temp_2)/(350-temp))
            pre_la.append(hh)

        else:
            ## no last output --
            prob_1 = sess.run(prob,feed_dict={embbed:m[j].reshape(1,350,4096),\
                          decoder_inputs:np.array(n[j]).reshape(1,351),\
                          decoder_outputs:np.array(l[j]).reshape(1,351),\
                          decoder_length: np.ones((1))*351,keep_prob:1})
            hh = np.argmax(prob_1[0],1)
            ans += list(hh)
            acc_1= sess.run(acc,feed_dict={embbed:m[j].reshape(1,350,4096),\
                          decoder_inputs:np.array(n[j]).reshape(1,351),\
                          decoder_outputs:np.array(l[j]).reshape(1,351),\
                          decoder_length: np.ones((1))*351,keep_prob:1})
            acc_trace.append(acc_1)
            pre_la.append(hh)
        count+=1
    opp = open(v_name[i] + '.txt','w')
    for i in ans:
    	opp.write(str(i)+'\n')
    acc_trace_1.append(np.mean(acc_trace))
    pre_trace_1.append(pre_la)
    
'''