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
import argparse

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Task_1')
	parser.add_argument('--train',type=bool,default=False,help='pure training')
	parser.add_argument('--self_train',type=bool,default=False,help='semi_supervised')
	parser.add_argument('--test',type=bool,default=False,help='test_phase')
	parser.add_argument('--train_path',type=str,help='training path')
	#parser.add_argument('--test_path',type=str,default='Fashion_MNIST_student/test/',help='testing path')
	#parser.add_argument('--train_path',type=str,help='training path')
	parser.add_argument('--test_path',type=str,help='testing path')
	parser.add_argument('--outfile',type=str,help='outfile')

	args = parser.parse_args()
	#print(args)
	import Task1
	model = Task1.task1(args)
	import Task1_train
	model_1 = Task1_train.task11(args)
	#print(args.train)
	#print(args.test)
	if args.train: 
		model_1.train()
	else : 
		model.test()


