import numpy as np
import random
import math
import sys,os,caffe

import adjust_layer

caffe_root = '/home/gql/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)

def adjust_alexnet(netinpath, netoutpath):
	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
	               caffe_root + netinpath,    
	              caffe.TEST);

	pufs_conv1 = np.load("/home/gql/Desktop/debug/pufs_conv1.npy");
	pufs_conv2 = np.load("/home/gql/Desktop/debug/pufs_conv2.npy");
	pufs_conv3 = np.load("/home/gql/Desktop/debug/pufs_conv3.npy");
	pufs_conv4 = np.load("/home/gql/Desktop/debug/pufs_conv4.npy");
	pufs_conv5 = np.load("/home/gql/Desktop/debug/pufs_conv5.npy");

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data


	aw_conv1 = np.zeros(np.shape(w_conv1));
	aw_conv2 = np.zeros(np.shape(w_conv2));
	aw_conv3 = np.zeros(np.shape(w_conv3));
	aw_conv4 = np.zeros(np.shape(w_conv4));
	aw_conv5 = np.zeros(np.shape(w_conv5));


	aw_conv1 = adjust_layer.adjust_layer(w_conv1, pufs_conv1, "conv1");
	aw_conv2 = adjust_layer.adjust_layer(w_conv2, pufs_conv2, "conv2");
	aw_conv3 = adjust_layer.adjust_layer(w_conv3, pufs_conv3, "conv3");
	aw_conv4 = adjust_layer.adjust_layer(w_conv4, pufs_conv4, "conv4");
	aw_conv5 = adjust_layer.adjust_layer(w_conv5, pufs_conv5, "conv5");


	net.params['conv1'][0].data[...] = aw_conv1;
	net.params['conv2'][0].data[...] = aw_conv2;
	net.params['conv3'][0].data[...] = aw_conv3;
	net.params['conv4'][0].data[...] = aw_conv4;
	net.params['conv5'][0].data[...] = aw_conv5;

   

	net.save(netoutpath);