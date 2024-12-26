import numpy as np
import random
import math
import sys,os,caffe

import adjust_layer
import generate_pufs
import puf_layer


caffe_root = '/home/gql/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)

if __name__=="__main__":
    #parameters
	netpufpath = 'models/bvlc_alexnet/bvlc_alexnet_puf.caffemodel'
	netguesspath = 'models/bvlc_alexnet/bvlc_alexnet_guess.caffemodel'



	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
               caffe_root + netpufpath,    
              caffe.TEST);

	pufs_conv1= np.load()
	pufs_conv1= np.load()
	pufs_conv1= np.load()
	pufs_conv1= np.load()
	pufs_conv1= np.load()

	pw_conv1 = net.params['conv1'][0].data
	pw_conv2 = net.params['conv2'][0].data
	pw_conv3 = net.params['conv3'][0].data
	pw_conv4 = net.params['conv4'][0].data
	pw_conv5 = net.params['conv5'][0].data

	gw_conv1 = puf_layer.puf_layer(w_conv1, pufs_conv1, ratio1, "conv1", "guess");
	gw_conv2 = puf_layer.puf_layer(w_conv2, pufs_conv2, ratio2, "conv2", "guess");
	gw_conv3 = puf_layer.puf_layer(w_conv3, pufs_conv3, ratio1, "conv3", "guess");
	gw_conv4 = puf_layer.puf_layer(w_conv4, pufs_conv4, ratio1, "conv4", "guess");
	gw_conv5 = puf_layer.puf_layer(w_conv5, pufs_conv5, ratio1, "conv5", "guess");

	net.params['conv1'][0].data[...] = gw_conv1;
	net.params['conv2'][0].data[...] = gw_conv2;
	net.params['conv3'][0].data[...] = gw_conv3;
	net.params['conv4'][0].data[...] = gw_conv4;
	net.params['conv5'][0].data[...] = gw_conv5;


	net.save(netguesspath);



