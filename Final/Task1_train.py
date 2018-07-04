import numpy as np 
#import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.contrib.layers as ly 
import os 
import sys
import pandas as pd 
import time
#from skimage import io
#from PIL import Image, ImageOps
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
#sklearn train_test validation

train_path = 'Fashion_MNIST_student'

class task1:
    def __init__(self,arg):
        #os.listdir(train_path)
        self.arg = arg
        train_data = self.data_loader(self.arg.train_path)

        self.test_data =self.test_loader(self.arg.test_path)
        print(len(self.test_data))
        #train_label = self.make_label(train_data)

        self.t_data ,self.v_data,self.t_label,self.v_label = train_test_split(train_data,train_label,random_state=9,test_size=0.05)
        self.build_model()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tensorboard_dir = 'Task_1/'   
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir) 
        print(self.arg.test)
        if self.arg.test : 
            #latest_checkpoint =tf.train.latest_checkpoint('model')
            self.saver.restore(self.sess,'99')
            print('load_model')
            #print(latest_checkpoint)


        one_hot = tf.one_hot(self.label,10)
        

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.prob,1),self.label),tf.float32))
        tf.summary.scalar('acc',self.acc)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot,logits=self.prediction))
        tf.summary.scalar('loss',self.loss)

        optimizer =tf.train.AdamOptimizer(0.0001)

        self.optimizer = optimizer.minimize(self.loss)
        self.merge = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(tensorboard_dir)     

        self.writer.add_graph(self.sess.graph)

    def data_loader(self,filepath):
        image_list = [] 
        temp = list(np.sort(os.listdir(filepath)))
        for i in range(len(temp)):
            temp_1 = os.path.join(filepath,temp[i])
            for j in os.listdir(temp_1):
                image_list.append(imread(os.path.join(temp_1,j))/255)
        return image_list
        
    def test_loader(self,filepath):
        image_list = [] 
        temp = list(np.sort(os.listdir(filepath)))
        for i in range(len(temp)):
            image_list.append(imread(os.path.join(filepath,str(i)+'.png'))/255)
        return image_list


    def make_label(self,train_data):
        each_len_label = len(train_data)/10 
        label = []

        for i in range(10):
            for _ in range(int(each_len_label)):
                label.append(i)

        return label

    def build_model(self):
        self.image = tf.placeholder(tf.float32 , shape = [None,28,28,1])
        self.label = tf.placeholder(tf.int64, shape = [None])

        num_units = [3,3,3]
        filter_size = [16,32,64]
        stride = [1,1,1]

        with tf.variable_scope('init'):
            x = ly.conv2d(self.image,16,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)

            for i in range(len(filter_size)):
                for j in range(len(num_units)):
                    if j==0:
                        if i ==0 : 
                            x = self.residual_unit(x,16,filter_size[i],stride[i])
                        else: 
                            x = self.residual_unit(x,filter_size[i-1],filter_size[i],stride[i])
                    else :
                        x = self.residual_unit(x,filter_size[i],filter_size[i],stride[i])
            x = tf.reduce_mean(x, [1, 2])
            #print(x)
            with tf.variable_scope('fully_connected'):
                #flat = ly.flatten(x)
                fc_1 = ly.fully_connected(x,256,activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm)
                fc_2 = ly.fully_connected(fc_1,256,activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm)
                self.prediction = ly.fully_connected(fc_2,10,activation_fn=None)
                self.prob = tf.nn.softmax(self.prediction)


    def residual_unit(self,input,in_filters,out_filters,stride,option=0):

        x = ly.conv2d(input,out_filters,kernel_size=3,stride=stride,padding='SAME',activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm)
        x = ly.conv2d(x,out_filters,kernel_size=3,stride=stride,padding='SAME',activation_fn=tf.nn.leaky_relu,normalizer_fn=ly.batch_norm)

        if in_filters != out_filters : 
            if option == 0 : 
                difference = out_filters - in_filters 
                left_pad = difference/2 
                right_pad = difference-left_pad 
                identity = tf.pad(input,[[0,0],[0,0],[0,0],[int(left_pad),int(right_pad)]])
                return x + identity
            else : 
                print('Not implement error')
                exit(1)
        else : 
            return x + input

    #def build_model(self):
    #   self.image - tf.placeholder(tf.float32,shape=[None,32,32,1])
    #   self.label = tf.placeholder(tf.int64,shape=[None])



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


    def next_batch(self,encoder_input,label,batch_size=128):
        le = len(encoder_input)
        epo = le//batch_size
        leftover = le - (epo*batch_size)
        for i in range(0,le,128):
            if i ==  (epo *batch_size) : 
                yield np.array(encoder_input[i:]+encoder_input[0:(batch_size-leftover)]) , np.array((label[i:]+label[0:(batch_size-leftover)]))
            else : 
                yield np.array(encoder_input[i:i+128]) , np.array(label[i:i+128])

    
    def train(self):
        if self.arg.train:
            epochs = 100
            for k in range(epochs):
                print(k)
                for i , j in self.next_batch(self.t_data,self.t_label):
                    #print(i.shape)
                    #print(j.shape)
                    feed_dict = {self.image:i.reshape(128,28,28,1),self.label:j}
                    merged,loss , acc , _ = self.sess.run([self.merge,self.loss,self.acc,self.optimizer],feed_dict=feed_dict)
                    print('{}/{} epochs. loss is {} ,acc is {}'.format(k+1,epochs,loss,acc))
                    self.writer.add_summary(merged)
                    if acc>0.7 : 
                        self.saver.save(self.sess,'model/'+str(k))
                if k% 5 == 0 : 
                    feed_dict = {self.image:np.array(self.v_data).reshape(100,28,28,1),self.label:np.array(self.v_label)}
                    acc = self.sess.run(self.acc,feed_dict=feed_dict)
                    print('*********Accuracy of validation set is {}.'.format(acc))



        if self.arg.self_train : 
            latest_checkpoint = tf.train.latest_checkpoint('model/')
            self.restore(self.sess,latest_checkpoint)
            epochs = 10 
            for k in range(epochs):
                for i , j in self.next_batch(self.test_data,self.prediction_label):
                    feed_dict = {self.image:i,self.label:j}
                    loss , acc , _ = self.sess.run([self.loss,self.acc,self.optimizer],feed_dict=feed_dict)
                    print('{}/{} epochs. loss is {} ,acc is {}'.format(k+1,epochs,loss,acc))
                    if acc>0.7 : 
                        self.saver.save(self.sess,'model/'+'self_t_'+str(k))
    

    def test(self):

            #latest_checkpoint = tf.train.latest_checkpoint('model/')
            #self.restore(self.sess,latest_checkpoint)
            test_batch = []
            #print(self.t_data[0])
            for i in range(0,len(self.test_data),128):
                test_batch.append(self.test_data[i:i+128])
            print(len(test_batch))
            epochs = 1
            prediction = []
            for i in test_batch:
                feed_dict = {self.image:np.array(i).reshape(-1,28,28,1)}
                probability = self.sess.run(self.prob,feed_dict=feed_dict)
                prediction+=list(np.argmax(probability,1))
                #print(probability)
            #print(len(prediction))
            np.save('pre_label.npy',prediction)
            Matrix = {}
            Matrix['image_id'] = [i for i in range(10000)]
            Matrix['predicted_label'] = prediction
            final = pd.DataFrame(Matrix)
            final.to_csv(self.arg.outfile,index=False)


                #print('{}/{} epochs. loss is {} ,acc is {}'.format(k+1,epochs,loss,acc))
                #if acc>0.7 : 
                    #self.saver.save(self.sess,'model/'+'self_t_'+str(k))