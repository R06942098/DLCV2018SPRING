import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow.contrib.layers as ly 
import tensorflow.contrib.rnn as rnn
import argparse
import os 
import sys 
import random
import time
#from skimage import io
#from PIL import Image, ImageOps
from scipy.misc import imread, imresize
#from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
#random.seed(9)

#### implement N way one shot learning .....!!!!!!
class Task2:
    def __init__(self,args):
        self.arg = args 
        self.K = 1 ## one shot first 
        self.fce = True 
        self.average_per_class_embeddings  = False

        #self.image_b = self.pre_train()
        self.image_n = self.pre_novel()
        #self.all_image = self.all_img()
        #print(len(self.image_n))
        #print(self.image_n[0][0].shape)
        self.image_test = self.pre_test()
        if self.arg.train:
            self.train_phase = True 
        else : 
            self.train_phase = False

        self.build_model()
        #self.build_model(layers=True,num=2)#,average_per_class_embeddings=True)
        #self.build_model()#self.average_per_class_embeddings=True)
        #print(self.g_variables)

        if self.fce : 
            var_list = self.g_variables +self.f_variables + self.cnn_variables 
        else : 
            var_list = self.cnn_variables

        for i in var_list : 
            print(i)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        optimizer = tf.train.AdamOptimizer(0.001,beta1=0.9)
        self.train_op = optimizer.minimize(self.loss,var_list=var_list)
        tf.summary.scalar('loss',self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.preds,1),self.target_label),tf.float32))
        tf.summary.scalar('acc',self.acc)
        self.sess.run(tf.global_variables_initializer())
        tensorboard_dir = 'Task_2/'   
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir) 

        self.writer = tf.summary.FileWriter(tensorboard_dir)
        self.writer.add_graph(self.sess.graph)

        if self.arg.test:
            #latestcheckpoint = tf.train.latestcheckpoint('model_1')
            #self.saver.restore(self.sess,'model_1/29')
            #self.saver.restore(self.sess,'model_1/33')
            #self.saver.restore(self.sess,'one_shot/59_224')
            self.saver.restore(self.sess,'59')
            print('Succesfully load model !!!!!')
        #print(self.image_embeddings)


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

    def maxpool(self,name,input_data,trainable=False):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding='SAME',name=name)
        return out 

    def conv(self,name,input_data,out_channel,parameters,trainable=None):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights',[3,3,in_channel,out_channel],dtype=tf.float32,trainable=trainable)
            biases = tf.get_variable('bias',[out_channel],dtype = tf.float32,trainable=trainable)
            conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding='SAME')
            res = tf.nn.bias_add(conv_res,biases)
            out = tf.nn.relu(res,name=name)
            parameters += [kernel,biases]
        return out , parameters

    def build_cnn(self,image,reuse=False):
        #num_units = [3 , 3, 3 ]
        #filter_size = [64,128,256]
        #stride = [1,1,1]
        num_units = [3 ]
        filter_size = [64]
        stride = [1]
        with tf.variable_scope('CNN_embedding') as scope:
            if reuse:
                scope.reuse_variables()
            x = ly.conv2d(image,16,kernel_size=3,stride=1,padding='SAME',activation_fn=tf.nn.leaky_relu)
            for i in range(len(filter_size)):
                for j in range(len(num_units)):
                    if j == 0 : 
                        if i == 0 : 
                            x = self.residual_unit(x,16,filter_size[i],stride[i])
                        else : 
                            x = self.residual_unit(x,filter_size[i-1],filter_size[i],stride[i])
                    else : 
                        x = self.residual_unit(x,filter_size[i],filter_size[i],stride[i])
            x = tf.reduce_mean(x,[1,2])
        self.cnn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='CNN_embedding')
        return x 

    def CNN_embedding(self,image,reuse=False):
        with tf.variable_scope('CNN_embedding') as scope:
            if reuse:
               scope.reuse_variables()
            outputs = image
            with tf.variable_scope('conv_layers'):
                for idx , num_filters in enumerate([64,64,64,64]):
                    with tf.variable_scope('g_conv_{}'.format(idx)):
                        if idx == len([64,64,64,64]) -1:  ## padding from valid to SAME !!
                            outputs = ly.conv2d(outputs,num_filters,kernel_size=3,stride=1,activation_fn=tf.nn.leaky_relu,padding='SAME')#,normalizer_fn=ly.batch_norm)
                        else : 
                            outputs = ly.conv2d(outputs,num_filters,kernel_size=3,stride=1,activation_fn=tf.nn.leaky_relu,padding='SAME')#,normalizer_fn=ly.batch_norm)

                        outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None,
                                                                       decay=0.99,
                                                                       scale=True, center=True
                                                                       ,is_training=self.train_phase)  

                        outputs = ly.max_pool2d(outputs,kernel_size=2,stride=2,padding='SAME') ##MAKE SURE IT 
            self.image_embeddings = ly.flatten(outputs)
        self.cnn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='CNN_embedding')
        return self.image_embeddings

    def lstm_cell(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(256) ## best : 256
        return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
    def _lstm_cell(self):

        lstm = tf.contrib.rnn.BasicLSTMCell(128) ## best : 128 
        return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)

    def g_embedding_biLSTM(self,inputs,reuse=False,layers=False,num=None):

        if layers : 
            layer_sizes = [32 for i in range(num)]
            #layer_sizes = [ 32 , 32 , 32 ,32]
        else : 
            layer_sizes = [ 32 ]

        with tf.variable_scope('encoder',reuse=reuse) as scope:
            if reuse:
               scope.reuse_variables()

            #fw_lstm_cells_encoder = [rnn.LSTMCell(num_units=layer_sizes[i], activation=tf.nn.tanh)
            #                             for i in range(len(layer_sizes))]
            #bw_lstm_cells_encoder = [rnn.LSTMCell(num_units=layer_sizes[i], activation=tf.nn.tanh)
            #                             for i in range(len(layer_sizes))]
           

            fw_lstm_cells_encoder = [self._lstm_cell() for i in range(len(layer_sizes))]
            bw_lstm_cells_encoder = [self._lstm_cell() for i in range(len(layer_sizes))]
            outputs ,outputs_state_fw , outputs_state_bw = rnn.stack_bidirectional_rnn(fw_lstm_cells_encoder,bw_lstm_cells_encoder,inputs,dtype=tf.float32)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder')

        return outputs
        #return tf.add(tf.stack(inputs),tf.stack(outputs))
    def f_embedding_biLSTM(self,support_emb,target_emb,K,reuse=False):

        layer_size = 64

        b, k, h_g_dim = support_emb.get_shape().as_list()

        b, h_f_dim = target_emb.get_shape().as_list()

        #layer_size = [100,100,100]
        with tf.variable_scope('F_emb') as scope:
            if reuse:
               scope.reuse_variables()
            #fw_lstm_cells_encoder = rnn.LSTMCell(num_units=layer_size, activation=tf.nn.tanh)
            fw_lstm_cells_encoder = self.lstm_cell()
            #cell_bw =  tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(3)])

            attentional_softmax = tf.ones(shape=(b, k)) * (1.0/k)
            h = tf.zeros(shape=(b, h_g_dim))
            c_h = (h, h)
            c_h = (c_h[0], c_h[1] + target_emb)
            reuse_1=False
            for i in range(K):
                attentional_softmax = tf.expand_dims(attentional_softmax, axis=2)
                attented_features = support_emb * attentional_softmax
                attented_features_summed = tf.reduce_sum(attented_features, axis=1)
                c_h = (c_h[0], c_h[1] + attented_features_summed)
                x, h_c = fw_lstm_cells_encoder(inputs=target_emb, state=c_h)
                #x , h_c = tf.nn.dynamic_rnn(cell_bw, inputs=target_emb, initial_state=c_h)
                attentional_softmax = tf.layers.dense(x, units=k, activation=tf.nn.softmax, reuse=reuse_1)
                reuse_1 = True

        outputs = x
        #self.f_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='F_emb')
        '''
        with tf.variable_scope('F_emb') as scope:
            if reuse:
               scope.reuse_variables()
            cell = self.lstm_cell()
            prev_state = cell.zero_state(100,tf.float32)
            for step in range(10):
                output , state = cell(target_emb,prev_state)
                h_k = tf.add(output,target_emb)
                content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1],support_emb))
                r_k = tf.reduce_sum(tf.multiply(content_based_attention,support_emb),axis=0)
                prev_state = rnn.LSTMStateTuple(state[0],tf.add(h_k,r_k))
        '''
        self.f_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='F_emb')

        return outputs


    def distance_net(self,support_set,input_img):
        reuse = False
        with tf.name_scope('distance_module'+'net') , tf.variable_scope('distance_module',reuse=reuse):
            eps = 1e-10
            similarities = []
            for support_image in tf.unstack(support_set,axis=0):
                sum_support = tf.reduce_sum(tf.square(support_image),1,keep_dims=True)
                support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support,eps,float("inf")))
                dot_product = tf.matmul(tf.expand_dims(input_img,1),tf.expand_dims(support_image,2))
                dot_product = tf.squeeze(dot_product,[1,])
                cosine_similarity = dot_product * support_magnitude
                similarities.append(cosine_similarity)
        similarities = tf.concat(axis=1,values=similarities)
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='distance_module')
        '''
        target_norm = input_img
        sup_similarity = []
        for i in tf.unstack(support_set):
            i_normed = tf.nn.l2_normalize(i,1)
            similarity = tf.matmul(tf.expand_dims(target_norm,1),tf.expand_dims(i_normed,2))
            sup_similarity.append(similarity)
        similarities = tf.squeeze(tf.stack(sup_similarity, axis=1))
        '''

        return similarities

    def attention_classifier(self,similarities,support_set_y):
        reuse = False 
        with tf.name_scope('attention-classifier-att'),tf.variable_scope('attention_classifier',reuse=reuse):
            #attention = tf.nn.softmax(similarities)
            preds = tf.squeeze(tf.matmul(tf.expand_dims(similarities,1),support_set_y))
            #preds = tf.squeeze(tf.matmul(tf.expand_dims(attention,1),support_set_y))
        self.a_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='attention_classifier')
        return preds

    def crossentropy_softmax(self, outputs, targets):
        normOutputs = outputs - tf.reduce_max(outputs, axis=-1)[:, None]
        logProb = normOutputs - tf.log(tf.reduce_sum(tf.exp(normOutputs), axis=-1)[:, None])
        return -tf.reduce_mean(tf.reduce_sum(targets * logProb, axis=1))

    def build_model(self,average_per_class_embeddings=False,layers=False,residual=False,num=None):
        self.target_image = tf.placeholder(tf.float32,shape=[100,32,32,3])  ##first one is batch size
        self.target_label = tf.placeholder(tf.int64,shape=[100])
        self.support_image = tf.placeholder(tf.float32,shape=[100,20*self.K,32,32,3])  ## 10 way -> 5 way 
        self.support_label = tf.placeholder(tf.int32,shape=[100,20*self.K])    ## 10 way -> 5 way 
        self.keep_prob = tf.placeholder(tf.float32)
        ### change to 20 way one shot 
        with tf.name_scope('losses'):
            self.support_set_labels = tf.one_hot(self.support_label,20) ###20 

            g_encoded_images = [] 

            count = 0
            for image in tf.unstack(self.support_image,axis=1):
                if count ==0 : 
                    if residual : 
                        support_cnn_embed = self.build_cnn(image)
                    else:
                        support_cnn_embed = self.CNN_embedding(image)
                    g_encoded_images.append(support_cnn_embed)
                    count+=1
                else:
                    if residual : 
                        support_cnn_embed = self.build_cnn(image,reuse=True)
                    else : 
                        support_cnn_embed = self.CNN_embedding(image,reuse=True)
                    g_encoded_images.append(support_cnn_embed)

            if self.average_per_class_embeddings : 
                g_encoded_images = tf.stack(g_encoded_images,axis=1)
                b , k , h = g_encoded_images.get_shape().as_list()
                g_encoded_images = tf.reshape(g_encoded_images,shape=[b,5,self.K,h])##20  ##100
                g_encoded_images = tf.reduce_mean(g_encoded_images,axis=2)


                self.support_set_labels = tf.reshape(self.support_set_labels,shape=[b,5,self.K,self.K]) ##20
                self.support_set_labels = tf.reduce_mean(self.support_set_labels,axis=2)

            if residual : 
                f_encoded_image = self.build_cnn(self.target_image,reuse=True)
            else: 
                f_encoded_image = self.CNN_embedding(self.target_image,reuse=True)

            if self.fce : ## Apply LSTM on embedding if fce is enables 
                g_encoded_images = self.g_embedding_biLSTM(g_encoded_images,layers=layers,num=num)
                f_encoded_images = self.f_embedding_biLSTM(tf.stack(g_encoded_images,axis=1),\
                                                            f_encoded_image,5)
                #f_encoded_images = self.f_embedding_biLSTM(g_encoded_images,f_encoded_image,10)
            g_encoded_images = tf.stack(g_encoded_images,axis=0)
            similarities = self.distance_net(g_encoded_images,f_encoded_images)

            self.preds = self.attention_classifier(similarities,self.support_set_labels)
            target_label = tf.one_hot(self.target_label,20)  #20

            self.loss = self.crossentropy_softmax(targets=target_label,outputs=self.preds)
            #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_label,logits=self.preds))
            #self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_label,logits=self.preds))

            #self.prob = tf.nn.softmax(preds)
    def data_aug(self):
        ### copy for 100 times 
        count = 0 
        data = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.09,horizontal_flip=True,zoom_range=0.09,shear_range=0.09,
            height_shift_range=0.09,fill_mode='nearest')
        #for batch in data.flow(img.reshape(1,64,64,3),batch_size=1):
        return data 


    def plot(self,x):
        x = x - np.min(x)
        x /= np.max(x)
        x *= 255  
        x= x.astype(np.uint8)
        x = x.reshape(32,32,3)
        return x 

    def pre_novel(self):
        target_path = self.arg.novel_path
        temp = list(np.sort(os.listdir(target_path)))
        random.seed(9)
        sample_id = random.sample([i for i in range(500)],1)  ## irugubak us fuve 
        #np.save('sample_1_shot.npy',sample_id)
        sample_id = np.load('sample_1_shot.npy')
        target_list = [[] for i in range(20)]
        count = 0
        for i in temp:
            temp_3 = os.path.join(target_path,i)
            temp_1 = list(np.sort(os.listdir(os.path.join(temp_3,'train'))))
            for j in sample_id:
                #print(j)
                x = imread(os.path.join(os.path.join(temp_3,'train'),temp_1[j]))/255
                #x = 2 *x - 1
                target_list[count].append(x)
            count+=1 


        ### i have remove /255 above 
        ### below augementation image to 500 pieces

        if self.arg.train : 
            data = self.data_aug()
            aug_image_n = [[] for i in range(20)]
            cla = 0 
            for i in target_list : 
                for img in i : 
                    count = 0 
                    for batch in data.flow(img.reshape(1,32,32,3),batch_size=1,save_to_dir='test_img',save_prefix='Face',save_format='png'):
                        x = self.plot(batch)/255 
                        x = 2*x - 1
                        aug_image_n[cla].append(x)
                        count +=1
                        if count == 500 :  ##100
                            break
                print(len(aug_image_n[cla]))
                cla+=1
            return aug_image_n
        else : 
            cla = 0 
            basic_sup = [[] for i in range(20)]
            for i in target_list:
                for img in i : 
                    #x = img/255 
                    x = 2*img -1 
                    basic_sup[cla].append(x)

                cla+=1 
            return basic_sup




        #lab = [0,10,23,30,32,35,48,54,57,59,60,64,66,69,71,82,91,92,93,95]

        #label_list = []
        #for i in lab : 
        #   for _ in range(K_shot):
        #       label_list.append(i)
        #print(target_list)
        return target_list 

    def pre_train(self):
        lab_1 = [0,10,23,30,32,35,48,54,57,59,60,64,66,69,71,82,91,92,93,95]
        lab = [] 
        for i in range(100):
            if i not in lab_1:
                lab.append(i)

        sup_list = [[] for i in range(80)]
        #sup_lab = [[] i for i in range(80)]
        temp = os.listdir(self.arg.train_path)
        count = 0
        for i in temp : 
            qq = os.listdir(os.path.join(self.arg.train_path,os.path.join(i,'train')))
            for j in qq : 
                x = imread(os.path.join(os.path.join(self.arg.train_path,os.path.join(i,'train')),j))/255
                x = 2 *x -1 
                sup_list[count].append(x)
            #for _ in range(500)
            #   sup_lab[count].append(lab[count])

            count +=1
        #print(sup_list)
        return sup_list

    def pre_test(self):
        test_path = self.arg.test_path
        test_list = [] 
        for i in range(2000):
            x = imread(os.path.join(test_path,str(i)+'.png'))/255
            x = 2*x - 1 
            test_list.append(x)
        return test_list

    '''

    def next_batch(self):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        for _ in range(100): 
            x_set = [] 
            y_set = []
            x = [] 
            y = []

            classes = np.random.permutation(80)[:19]

            for i , c  in enumerate(classes):
                samples = np.random.permutation(500)[:self.K]
                for s in samples : 
                    x_set.append(self.image_b[c][s])
                    y_set.append(i)

            target_ind = random.sample([i for i in range(20)],1)[0]
            x_hat_batch.append(self.image_n[target_ind][:self.K][0])
            x_set.append(self.image_n[target_ind][:self.K][0])
            #print(self.image_n[target_ind][:self.K].shape)
            #print(x_hat_batch[0][0].shape)
            y_hat_batch.append(19) 
            y_set.append(19)

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)  

        #return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)
        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)

    '''
    def all_img(self):
        lab_1 = [0,10,23,30,32,35,48,54,57,59,60,64,66,69,71,82,91,92,93,95]
        all_image_list = []

        count_n = 0
        count_b = 0
        for i in range(100):
            if i in lab_1 : 
                all_image_list.append(self.image_n[count_n])
                count_n+=1
            else:
                all_image_list.append(self.image_b[count_b])
                count_b+=1
        return all_image_list
    '''
    def next_batch(self):
        temp = self.image_b + self.image_n
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        for _ in range(100):
            x_set = []
            y_set = []
            x = []
            y = []
            classes = np.random.permutation(100)[:20]
            target_class = np.random.randint(20)
            for i, c in enumerate(classes):
                samples = np.random.permutation(10)[:self.K+1]
                for s in samples[:-1]:
                    x_set.append(temp[c][s])
                    y_set.append(i)

                if i == target_class:
                    x_hat_batch.append(temp[c][samples[-1]])
                    y_hat_batch.append(i)

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)
    '''

    def next_batch(self,par=None):

        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        if par == 1 :
            (a,b) = (0,50)
        else : 
            (a,b) = (50,100)

        for num in range(100):
            x_set = []
            y_set = []
            x = []
            y = []
            total_class = [i for i in range(100)]
            #selected_classes = np.random.permutation(100)[:20]
            episode_labels = [i for i in range(20)]
            #class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
            #                      zip(selected_classes, episode_labels)}
            #target_class = random.sample(list(selected_classes),1)[0]
            target_class = num 
            total_class.pop(target_class)
            selected_classes = list(np.random.permutation(total_class)[:19])
            selected_classes.append(target_class)
            random.shuffle(selected_classes)
            class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}
            #print(len(selected_classes))
            for class_sample in selected_classes:

                if len(self.all_image[class_sample])==500:

                    choose_sample = random.sample([i for i in range(500)],1) #5
                else : 
                    choose_sample = random.sample([i for i in range(5)],1) #5

                for sample in choose_sample:

                    x_set.append(self.all_image[class_sample][sample])
                    y_set.append(int(class_to_episode_label[class_sample]))

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

            if len(self.all_image[target_class]) ==500 : 

                target_sample = random.sample([i for i in range(500)],1)[0]

            else :

                target_sample = random.sample([i for i in range(5)],1)[0]

            x_hat_batch.append(self.all_image[target_class][target_sample])
            y_hat_batch.append(int(class_to_episode_label[target_class]))

        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)

    def test_next_batch(self,num=0,cls_num=None):
        temp = self.image_test
        temp_1 = self.image_n
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []

        for _ in range(100):#100

            x_set = []
            y_set = []

            x = []
            y = []

            #classes = np.random.permutation(100)[:20]

            classes = np.array([i for i in range(20)])
            #classes = classes[cls_num:cls_num+5]
            #target_class = np.random.randint(20)
            for i , c  in enumerate(classes):
                #for ic in classes:
                samples = np.random.permutation(1)[:self.K]##5
                for s in samples : 
                    x_set.append(self.image_n[c][s])
                    y_set.append(i)
            #for i, c in enumerate(classes):

            #    samples = np.random.permutation(15)[:self.K]
            #    for s in samples[:-1]:
            #        x_set.append(temp_1[c][s])
            #        y_set.append(i)


            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

            x_hat_batch.append(temp[num])
            num+=1
            y_hat_batch.append(i)
        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)

    def train(self):
        temp = np.load('theta_cnn_1.npy')
        assign_op = []
        for i in range(len(temp)):
            assign_op.append(tf.assign(self.cnn_variables[i],temp[i]))
        #self.sess.run(assign_op)
        epochs_acc = []
        epochs_loss = []
        epochs=60
        for i in range(epochs):
            temp_loss = []
            temp_acc = []
            for step in range(100):
                    #for y in [1,2]:
                    x_set, y_set, x_hat, y_hat = self.next_batch()
                    #print(x_set.shape)
                    #print(x_hat.shape)
                    feed_dict = {self.target_image:x_hat,self.target_label:y_hat,self.support_image:x_set,self.support_label:y_set,self.keep_prob:0.7} ##0.7
                    acc_1 , loss_1 , _ = self.sess.run([self.acc,self.loss,self.train_op],feed_dict=feed_dict)
                    #print(self.sess.run(self.preds,feed_dict=feed_dict))
                    print('********** {}/{} . Loss is {}. Acc is {}.'.format(i+1,epochs,loss_1,acc_1))
                    temp_acc.append(acc_1)
                    temp_loss.append(loss_1)
            epochs_loss.append(np.mean(temp_loss))
            epochs_acc.append(np.mean(temp_acc))
            if acc_1 > 0.5 : 
                self.saver.save(self.sess,'model_1/'+str(i))
        np.save('epochs_acc.npy',epochs_acc)
        np.save('epochs_loss.npy',epochs_loss)



    def test(self):
        lab = [00,10,23,30,32,35,48,54,57,59,60,64,66,69,71,82,91,92,93,95]
        cls_num = [0,5,10,15]
        #temp = [i for i in range(0,2000,100)]
        temp = [i for i in range(0,2000,100)]
        ans_list = []
        for i in temp:
                #count = 0 
                #for qq in cls_num:
                x_set, y_set, x_hat, y_hat = self.test_next_batch(num=i)
                #x_set, y_set, x_hat, y_hat = self.next_batch()
                #x_hat = x_hat.reshape(100,32,32,3)
                x_hat = x_hat.reshape(100,32,32,3)
                feed_dict = {self.target_image:x_hat,self.target_label:y_hat,self.support_image:x_set,self.support_label:y_set,self.keep_prob:1}
                preds = self.sess.run(self.preds,feed_dict=feed_dict)


                #if count == 0 : 
                #    ex_l = preds
                #    count+=1
                #else : 
                #    ex_l = np.concatenate((ex_l,preds),axis=1)
                #print(preds.shape)
            

                #feed_dict = {self.target_image:self.image_test[0].reshape(1,32,32,3),self.support_set:x_set}
                #preds = self.sess.run([self.preds],feed_dict=feed_dict)
                #print(preds.shape)
                #print(preds)
                #print(acc_1)
            #ans_list += list(np.argmax(preds,1))
            #print(ex_l.shape)
                print(preds.shape)
                ans_list += list(np.argmax(preds,1))
        print(len(ans_list))
        print(ans_list)
        final_ans = [] 
        for i in ans_list:
            final_ans.append(lab[i])
        Matrix = {}
        Matrix['image_id'] = [i for i in range(2000)]
        Matrix['predicted_label'] = final_ans
        final = pd.DataFrame(Matrix)
        final.to_csv(self.arg.outfile,index=False)


    def ensemble(self): 
        model_file = ['44','59','59_33','59_40','64_392','69_38','54_381','49_398','49_351'] ##'99_341 , '69_375' ,'54_40'is bi-lstm (2 layers)
        ans = []

        for i in model_file : 
            self.saver.restore(self.sess,i)
            ans_list= [] 
            temp = [i for i in range(0,2000,100)]
            ans_list = []
            for i in temp:
                x_set, y_set, x_hat, y_hat = self.test_next_batch(num=i)
                #x_set, y_set, x_hat, y_hat = self.next_batch()
                x_hat = x_hat.reshape(100,32,32,3)

                feed_dict = {self.target_image:x_hat,self.target_label:y_hat,self.support_image:x_set,self.support_label:y_set,self.keep_prob:1}
                preds = self.sess.run(self.preds,feed_dict=feed_dict)  
                ans_list.append(preds)
            ans.append(np.array(ans_list).reshape(2000,20))

        model_file = ['69_375' ,'59_394','54_40' ,'49_425','59_432','59_unknown']
        number = [ 2 ,2, 1 , 1, 1,2 ]

        count = 0 
        for i , j in zip(model_file,number) : 
            print(j)
            #print(count)
            tf.reset_default_graph()
            self.build_model(layers=True,num=j) 
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess,i)

            ans_list= [] 
            temp = [i for i in range(0,2000,100)]

            ans_list = []
            for i in temp:
                x_set, y_set, x_hat, y_hat = self.test_next_batch(num=i)
                #x_set, y_set, x_hat, y_hat = self.next_batch()
                x_hat = x_hat.reshape(100,32,32,3)

                feed_dict = {self.target_image:x_hat,self.target_label:y_hat,self.support_image:x_set,self.support_label:y_set,self.keep_prob:1}
                preds = self.sess.run(self.preds,feed_dict=feed_dict)  
                ans_list.append(preds)

            ans.append(np.array(ans_list).reshape(2000,20))

        # tf.reset_default_graph()    ### add for three bilstm ..... (['49_425']) #four ['59_432']
        #self.build_model(layers=False , residual=True)
        #  


        final_ans = []
        #for index in range(2000): 
        for i in range(len(ans)):
            if i == 0 : 
                count = 0 
                count += ans[i]
            count +=ans[i]
        print(count)
        print(count.shape)

        ans = list(np.argmax(count,1))
        print(ans)
        lab = ['00',10,23,30,32,35,48,54,57,59,60,64,66,69,71,82,91,92,93,95]
        final_ans = [] 
        for i in ans:
            final_ans.append(lab[i])
        Matrix = {}
        Matrix['image_id'] = [i for i in range(2000)]
        Matrix['predicted_label'] = final_ans
        final = pd.DataFrame(Matrix)
        final.to_csv('pre_1.csv',index=False)


###   test for one shot now !!!