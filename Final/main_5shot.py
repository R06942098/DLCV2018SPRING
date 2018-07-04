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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Task_2')
	parser.add_argument('--train',type=bool,default=False,help='pure training')
	parser.add_argument('--test',type=bool,default=False,help='test_phase')
	parser.add_argument('--novel_path',type=str,help='training path')
	#parser.add_argument('--train_path',type=str,default='task2-dataset/base',help='testing path')
	parser.add_argument('--test_path',type=str,help='testing path')
	parser.add_argument('--ensemble',type=bool,default=False,help='ensemble?')
	parser.add_argument('--way',type=bool,default=False,help='ensemble?')
	parser.add_argument('--outfile',type=str,help='ensemble?')

	args = parser.parse_args()
	#print(args)

	import five_shot
	model = five_shot.Task2(args)
	#print(args.train)
	if args.train: 
		model.train()
	elif args.test : 
		model.test()
	else : 
		model.ensemble()
	
	'''
	
	import Task2_1

	model = Task2_1.Task2(args)

	if args.way:
		model.train()
	
	if args.test : 
		model.test()
	'''