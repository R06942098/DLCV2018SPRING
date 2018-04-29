import tensorflow as tf
import os 
import pandas as pd 
import cv2 
import numpy as np
from scipy.misc import imread, imresize
from skimage import io 
import matplotlib.pyplot as plt 
from math import ceil
import scipy.misc
import argparse
import sys

### $1 : training folder 
### $2 : validation folder 
### $3 : output folder 


label_chinese = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unkonown']
label_value = [(0,255,255),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]

def preprocess(filepath):
    temp = os.listdir(filepath)
    pic = []
    mask = []
    for i in temp : 
        if i[5:9] =='mask' : 
            mask.append(i)
        if i[5:8] =='sat':
            pic.append(i)
    gg = []
    qq = []
    for i in pic : 
        ls = io.imread(filepath+'/'+i).astype(np.float32)
        gg.append(ls)
    for i in mask: 
        ls = io.imread(filepath+'/'+i).astype(np.float32)
        qq.append(ls)
    return np.array(gg) , np.array(qq)

def add_label(matrix,label_value):
    te = np.zeros([512,512,7])
    a = matrix[:,:,0]
    b = matrix[:,:,1]
    c = matrix[:,:,2]
    for i in range(512):
        for j in range(512):
            temp = (a[i,j],b[i,j],c[i,j])
            for k in range(7):
                if temp ==label_value[k] : 
                    te[i,j,k]=1
    return te    

def map_la(matrix,label_value):
    temp = np.zeros([512,512])
    a = matrix[:,:,0]
    b = matrix[:,:,1]
    c = matrix[:,:,2]
    for i in range(512):
        for j in range(512):
            te = (a[i,j],b[i,j],c[i,j])
            for k in range(7):
                if te ==label_value[k] : 
                    temp[i,j]=k
    temp = temp.reshape(512,512,1)
    return temp
            
'''     
def load_data():
    pic ,mask = preprocess()
    mask_label = np.zeros((2313,512,512,1))
    for i in range(len(mask)):
        mask_label[i]= map_la(mask[i],label_value)    
    return pic , mask_label
'''       

t_data , t_1 = preprocess(sys.argv[1])

t_label = []
for i in t_label :
    cc = map_la(i,label_value)
    t_label.append(c)

t_label = np.array(t_label)

#t_label = np.load('label.npy')


VGG_MEAN = [103.939, 116.779, 123.68]

def maxpool(name,input_data,trainable=False):
    out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding='SAME',name=name)
    return out 

weight_file = 'vgg16_weights.npz'

parameters = [] 


### 放在sess裡面就行了####



def load_weights(weight_file, sess,parameters):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if i < 26:
            print(i, k, np.shape(weights[k]))
            sess.run(parameters[i].assign(weights[k]))


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


def conv_7(name,input_data,out_channel,trainable=None):
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',[7,7,in_channel,out_channel],dtype=tf.float32,trainable=trainable)
        biases = tf.get_variable('bias',[out_channel],dtype = tf.float32,trainable=trainable)
        conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding='SAME')
        res = tf.nn.bias_add(conv_res,biases)
        out = tf.nn.relu(res,name=name)
    return out

def conv_1(name,input_data,out_channel,relu=False,trainable=None):
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',[1,1,in_channel,out_channel],dtype=tf.float32,trainable=trainable)
        biases = tf.get_variable('bias',[out_channel],dtype = tf.float32,trainable=trainable)
        conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding='SAME')
        res = tf.nn.bias_add(conv_res,biases)
        if relu :
            out = tf.nn.relu(res,name=name)
        else : 
            out = res
    return out


def return_la(matrix,label_value):
    ## input為2維矩陣
    a = np.zeros([512,512])
    b = np.zeros([512,512])
    c = np.zeros([512,512])
    for i in range(512):
           for j in range(512):
               temp = matrix[i,j]
               z=label_value[temp]
               a[i,j]=z[0]
               b[i,j]=z[1]
               c[i,j]=z[2]
    return np.stack([a,b,c],axis=2)

def pre_val(filepath):
    temp = os.listdir(filepath)
    pic = []
    mask = []
    for i in temp : 
        if i[5:9] =='mask' : 
            mask.append(i)
        if i[5:8] =='sat':
            pic.append(i)
    gg = []
    qq = []
    for i in pic : 
        ls = io.imread(filepath+'/'+i).astype(np.float32)
        gg.append(ls)
    for i in pic : 
        ls = io.imread(filepath+'/'+i).astype(np.float32)
        qq.append(ls)
    return np.array(gg) 

tf.reset_default_graph()

img = tf.placeholder(tf.float32,shape=[None,512,512,3],name='input_image')
keep_prob = tf.placeholder(tf.float32)


lab = tf.placeholder(tf.int32,shape=[None,512,512,1],name='annotation')
r , g , b = tf.split(img,3,3)

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

conv6 = conv_7('conv6_1',pool5,4096,trainable=True)
conv6 = tf.nn.dropout(conv6,keep_prob=keep_prob)



def _variable_with_weight_decay(shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        return var

def _score_layer(name , bottom , num_classes=7):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        num_input = in_features
        stddev = (2 / num_input)**0.5
        # Apply convolution
        w_decay = 5e-4
        weights = _variable_with_weight_decay(shape, stddev, w_decay)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = tf.get_variable(name='biases', shape=[num_classes],initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, conv_biases)

        return bias
    
def get_deconv_filter(name,f_shape): 
    width = f_shape[0]
    height = f_shape[1]
    f = ceil(width/2)
    c = (2*f-1-f%2)/(2.0*f)
    bilinear = np.zeros([f_shape[0],f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1-abs(x/f-c)) * (1-abs(y/f-c))
            bilinear[x,y]=value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:,:,i,i] = bilinear
    init = tf.constant_initializer(value=weights,dtype=tf.float32)

    return tf.get_variable(name,initializer=init,shape=weights.shape)


def _upscore_layer(bottom, shape,
                       num_classes, name,ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
                # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, num_classes, in_features]

            # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        weights = get_deconv_filter('up_filter',f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

    return deconv

conv7 = conv_1('conv7_1',conv6,4096,relu=True,trainable=True)
conv7 = tf.nn.dropout(conv7,keep_prob=keep_prob)

#conv8 = _upscore_layer(conv7,7,trainable=True)
conv8 = _score_layer('conv8_1',conv7,7)

pred = tf.argmax(conv8,dimension=3)

upscore = _upscore_layer(conv8,shape=tf.shape(bgr),num_classes=7,ksize=64,stride=32,name='up_score')

#epsilon = tf.constant(value=1e-4)

#softmax = tf.nn.softmax(logits) + epsilon

pred_up = tf.argmax(upscore,dimension=3)

pred_ex = tf.expand_dims(pred_up, dim=3)


num_classes = 7




val  = pre_val(sys.argv[2])



def next_batch(x,y,batch_size=32):
    temp = np.arange(len(x))
    np.random.shuffle(temp)
    x = x[temp]
    y = y[temp]
    le = len(x)
    epo = le//batch_size
    for i in range(0,batch_size*epo,32):
        yield x[i:i+32] , y[i:i+32]

tensorboard_dir = 'tensorboard/'   # 保存目录
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)




writer = tf.summary.FileWriter(tensorboard_dir)      



with tf.name_scope('loss') :
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=upscore,labels=tf.squeeze(lab,3)))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    
saver = tf.train.Saver()
sess = tf.Session()
initializer = tf.global_variables_initializer()
sess.run(initializer)
bb = load_weights(weight_file,sess,parameters)
writer.add_graph(sess.graph)


process_dir = 'training_process/'   # 保存目录
if not os.path.exists(process_dir):
    os.makedirs(process_dir)


test = np.array([val[9],val[98],val[108]])
loss_trace = []
for i in range(50): 
    print(i)
    epoch_loss = []
    for batch_x,batch_y in next_batch(t_data,t_label):
        loss_1,_ = sess.run([loss,optimizer],feed_dict={img:batch_x,lab:batch_y,keep_prob:0.5})
        epoch_loss.append(loss_1)
    loss_trace.append(np.mean(epoch_loss))
    if i% 5 ==0:
        print(np.mean(epoch_loss))
    if i% 6 == 0 : 
        qq = sess.run(pred_up,feed_dict={img:test,keep_prob:1})
        count=0
        for k in qq : 
            jk = k.reshape(512,512)
            temp_3 = return_la(jk,label_value)
            temp_3 = temp_3.astype(np.uint8)
            io.imsave('training_process/'+str(i)+'_'+str(count)+'mask.png',temp_3)
            count+=1
            
saver.save(sess,'temp/fc8.ckpt')
       
bat = []
for i in range(0,len(val),32):
    bat.append(val[i:i+32])



'''
val_1= val[0:32]
val_2 =val[32:64]
val_3 = val[64:96]
val_4 = val[96:128]
val_5 = val[128:160]
val_6 = val[160:192]
val_7 = val[192:224]
val_8 = val[224:]

bat = [val_1,val_2,val_3,val_4,val_5,val_6,val_7,val_8]

'''
answer= []


for i in bat:
    qq = sess.run(pred_up,feed_dict={img:i,keep_prob:1})
    answer.append(qq)

answer_dir = sys.argv[3]  # 保存目录
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

count=0
for i in answer :
    for j in i :
        jk = j.reshape(512,512)
        temp_3 = return_la(jk,label_value)
        temp_3 = temp_3.astype(np.uint8)
        temp = 4-len(str(count))
        fin = '0'*temp + str(count)
        io.imsave(sys.argv[3]+fin+'_mask.png',temp_3)
        count+=1


def read_masks_jpg(filepath):
    '''
    Read masks from directory and tranform to cate gorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def read_masks_val(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file[5:9]=='mask']
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

label_a = read_masks_val(sys.argv[2])
pred = read_masks_jpg(sys.argv[3])
score = mean_iou_score(pred,label_a)
