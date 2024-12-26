import numpy as np
import random
import math
import sys,os,caffe

import puf_layer



caffe_root = '/home/gql/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)

def puf_alexnet(netinpath, netoutpath, pufs_conv1, pufs_conv2, pufs_conv3, pufs_conv4, pufs_conv5, ratio1, ratio2, operation):
	print "path"
	print ("netinpath", netinpath);
	print ("netoutpath", netoutpath);

	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
	               caffe_root + netinpath,    
	              caffe.TEST);
	print "path"
	print ("netinpath", netinpath);
	print ("netoutpath", netoutpath);

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data


	pw_conv1 = np.zeros(np.shape(w_conv1));
	pw_conv2 = np.zeros(np.shape(w_conv2));
	pw_conv3 = np.zeros(np.shape(w_conv3));
	pw_conv4 = np.zeros(np.shape(w_conv4));
	pw_conv5 = np.zeros(np.shape(w_conv5));

	pw_conv1 = puf_layer.puf_layer(w_conv1, pufs_conv1, ratio1, "conv1", operation);
	pw_conv2 = puf_layer.puf_layer(w_conv2, pufs_conv2, ratio2, "conv2", operation);
	pw_conv3 = puf_layer.puf_layer(w_conv3, pufs_conv3, ratio1, "conv3", operation);
	pw_conv4 = puf_layer.puf_layer(w_conv4, pufs_conv4, ratio1, "conv4", operation);
	pw_conv5 = puf_layer.puf_layer(w_conv5, pufs_conv5, ratio1, "conv5", operation);

	net.params['conv1'][0].data[...] = pw_conv1;
	net.params['conv2'][0].data[...] = pw_conv2;
	net.params['conv3'][0].data[...] = pw_conv3;
	net.params['conv4'][0].data[...] = pw_conv4;
	net.params['conv5'][0].data[...] = pw_conv5;

	net.save(netoutpath);