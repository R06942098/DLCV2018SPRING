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
from sklearn.manifold import TSNE
import tensorflow.contrib.layers as ly 


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

#f_path = file_path = 'HW5_data/TrimmedVideos/video/valid'
#validation_csv = pd.read_csv(sys.argv[3])
validation_csv = pd.read_csv(sys.argv[2])

validation_categorical = list(validation_csv[['Video_category']].values.reshape(-1))
validation_name = list(validation_csv[['Video_name']].values.reshape(-1))
#validation_label = list(validation_csv[['Action_labels']].values.reshape(-1))

validation_list = []
for i in range(len(validation_name)):
    vv = readShortVideo(sys.argv[1],validation_categorical[i],validation_name[i])
    validation_list.append(vv)    



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


conv1_1 ,parameters= conv('conv1re_1',bgr,64,parameters,trainable=True)
conv1_2 , parameters = conv('conv1_2',conv1_1,64,parameters,trainable=True)
pool1 = maxpool('poolre1',conv1_2,trainable=False)


conv2_1 , parameters = conv('conv2_1',pool1,128, parameters,trainable=True)
conv2_2 , parameters = conv('conwe2_2',conv2_1,128, parameters,trainable=True)
pool2 = maxpool('pool2',conv2_2,trainable=False)


conv3_1 , parameters = conv('conv3_1',pool2,256,parameters,trainable=True)
conv3_2 , parameters = conv('convrwe3_2',conv3_1,256,parameters,trainable=True)
conv3_3 , parameters = conv('convrwe3_3',conv3_2,256,parameters,trainable=True)
pool3 = maxpool('poolre3',conv3_3,trainable=False)


conv4_1 , parameters  = conv('conv4_1',pool3,512,parameters,trainable=True)
conv4_2 , parameters = conv('convrwe4_2',conv4_1,512,parameters,trainable=True)
conv4_3 , parameters = conv('convrwe4_3',conv4_2,512,parameters,trainable=True)
pool4 = maxpool('pool4',conv4_3,trainable=False)


conv5_1 , parameters  = conv('conv5_1',pool4,512,parameters,trainable=True)
conv5_2 , parameters  = conv('convrwe5_2',conv5_1,512,parameters,trainable=True)
conv5_3 , parameters  = conv('convrwe5_3',conv5_2,512,parameters,trainable=True)
pool5 = maxpool('pool5',conv5_3,trainable=True)

fc_6 , parameters = fc_layer(pool5,'fc1',parameters,trainable=True)

#fc_7 , parameters = fc_layer(fc_6,'fc_7',parameters,trainable=False)

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


emb = []
for i in validation_list :
    temp = []
    for j in i : 
        q =sess.run(fc_6,feed_dict = {img:j})
        temp.append(q)
    emb.append(np.array(temp))

emb_v = [] 
for i in emb:
    count = 0 
    dd = [0,i.shape[0]//2,-1]
    for j in dd: 
        if count == 0:
            temp = i[j]
            count+=1
            continue
        temp = np.concatenate((temp,i[j]),1)
    emb_v.append(temp.reshape(-1))


#tf.reset_default_graph()

embbed = tf.placeholder(tf.float32,shape=[None,12288])

lab = tf.placeholder(tf.int64,shape=[None])

logit = ly.fully_connected(embbed,4096,activation_fn=tf.nn.relu)

logit = ly.fully_connected(logit,1024,activation_fn=tf.nn.relu)

logit = ly.fully_connected(logit,11,activation_fn=None)

prob = tf.nn.softmax(logit)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=tf.one_hot(lab,11)))

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob,1),lab),tf.float32))

optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)


def next_batch(input_image,label , batch_size=64):
    le = len(input_image)
    #c = np.arange([i for i in range(le)])
    epo = le//batch_size
    #last  = epo*batch_size
    #random.shuffle(input_image)
    #for i in range(0,batch_size*epo,16):
    #    yield np.array(encoder_input[i:i+16])
    for i in range(0,le,64):
        if i == epo *batch_size :
            yield np.array(input_image[i:]) , np.array(label[i:])
        else :
            yield np.array(input_image[i:i+64]) , np.array(label[i:i+64])
            

sess= tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

saver.restore(sess,'150epochs')

for i in range(1):
    acc_trace=[]
    pre_trace = []
    for m,n in next_batch(emb_v,validation_label):
        #acc_1= sess.run(acc,feed_dict={embbed:m,lab:n})
        prob_1= sess.run(prob,feed_dict={embbed:m,lab:n})
        pre_trace += list(np.argmax(prob_1,1))
        #merge = sess.run(merged,feed_dict={embbed:m,lab:n,keep_prob:1.0})
        #acc_trace.append(acc_1)
        #writer.add_summary(merge,i)
    #print(np.mean(acc_trace))
    #merge = sess.run(merged,feed_dict={embbed:m,lab:n,keep_prob:1.0})

pp = open(os.path.join(sys.argv[3],'p1_result.txt'),'w')
for i in pre_trace: 
    pp.write(str(i)+'\n')
 

'''

X_embedded = TSNE(n_components=2,random_state=9).fit_transform(np.array(emb_v))


color = ['b','g','r','c','m','y','green','tomato']
shape = ["indigo",'C2','C5']
for i , j ,k  in zip(X_embedded[:,0],X_embedded[:,1],validation_label):
    if k <= 7 : 
        plt.scatter(i,j,s=1.5,color = color[k])
    else : 
        plt.scatter(i,j,s=1.5,color=shape[10-k])
        
plt.savefig(os.path.join(sys.argv[3],'CNN_feature.png'))
'''


#for i in range(0,len(emb_v),64):
    
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=embe_1,labels=lab))

#acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob,1),lab),tf.float32))

#optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)


