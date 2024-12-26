import numpy as np
import random
import math
import sys,os,caffe

import baseweight

caffe_root = '/home/gql/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
caffe.set_mode_gpu()


def base_alexnet(netinpath, netoutpath):
	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
           caffe_root + netinpath,    
          caffe.TEST);

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data


	shape1 = np.shape(w_conv1);
	shape2 = np.shape(w_conv2);
	shape3 = np.shape(w_conv3);
	shape4 = np.shape(w_conv4);
	shape5 = np.shape(w_conv5);

	for i in range (0, shape1[0]):
		for j in range (0, shape1[1]):
			for k in range (0, shape1[2]):
				for l in range (0, shape1[3]):
					w_conv1[i][j][k][l] = baseweight.baseweight(w_conv1[i][j][k][l]);

	for i in range (0, shape2[0]):
		for j in range (0, shape2[1]):
			for k in range (0, shape2[2]):
				for l in range (0, shape2[3]):
					w_conv2[i][j][k][l] = baseweight.baseweight(w_conv2[i][j][k][l]);

	for i in range (0, shape3[0]):
		for j in range (0, shape3[1]):
			for k in range (0, shape3[2]):
				for l in range (0, shape3[3]):
					w_conv3[i][j][k][l] = baseweight.baseweight(w_conv3[i][j][k][l]);

	for i in range (0, shape4[0]):
		for j in range (0, shape4[1]):
			for k in range (0, shape4[2]):
				for l in range (0, shape4[3]):
					w_conv4[i][j][k][l] = baseweight.baseweight(w_conv4[i][j][k][l]);
	for i in range (0, shape5[0]):
		for j in range (0, shape5[1]):
			for k in range (0, shape5[2]):
				for l in range (0, shape5[3]):
					w_conv5[i][j][k][l] = baseweight.baseweight(w_conv5[i][j][k][l]);



	net.params['conv1'][0].data[...] = w_conv1;
	net.params['conv2'][0].data[...] = w_conv2;
	net.params['conv3'][0].data[...] = w_conv3;
	net.params['conv4'][0].data[...] = w_conv4;
	net.params['conv5'][0].data[...] = w_conv5;

	net.save(netoutpath);	