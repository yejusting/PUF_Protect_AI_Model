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
	netoripath = 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
	net_ori_recover_path = 'models/bvlc_alexnet/ori_recover_model.caffemodel'
	net_adj_recover_path = 'models/bvlc_alexnet/adj_recover_model.caffemodel'

    # can be set
	bitlength = 12
	errorrate = 0.08
	ratio1 = 0.9;
	ratio2 = 0.96;
	lockrate = 0.8


	pufs_conv1 = generate_pufs.generate_pufs(lockrate, 16, 3, bitlength, errorrate);
	pufs_conv2 = generate_pufs.generate_pufs(lockrate, 2, 16, bitlength, errorrate);
	pufs_conv3 = generate_pufs.generate_pufs(lockrate, 3, 16, bitlength, errorrate);
	pufs_conv4 = generate_pufs.generate_pufs(lockrate, 3, 12, bitlength, errorrate);
	pufs_conv5 = generate_pufs.generate_pufs(lockrate, 2, 16, bitlength, errorrate);

	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
               caffe_root + netoripath,    
              caffe.TEST);

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data

	print "puf"
	pw_conv1 = np.zeros(np.shape(w_conv1));
	pw_conv2 = np.zeros(np.shape(w_conv2));
	pw_conv3 = np.zeros(np.shape(w_conv3));
	pw_conv4 = np.zeros(np.shape(w_conv4));
	pw_conv5 = np.zeros(np.shape(w_conv5));

	pw_conv1 = puf_layer.puf_layer(w_conv1, pufs_conv1, ratio1, "conv1", "reliable");
	pw_conv2 = puf_layer.puf_layer(w_conv2, pufs_conv2, ratio2, "conv2", "reliable");
	pw_conv3 = puf_layer.puf_layer(w_conv3, pufs_conv3, ratio1, "conv3",  "reliable");
	pw_conv4 = puf_layer.puf_layer(w_conv4, pufs_conv4, ratio1, "conv4", "reliable");
	pw_conv5 = puf_layer.puf_layer(w_conv5, pufs_conv5, ratio1, "conv5", "reliable");
	print "puf end"

	print "recover unreliable"
	rw_conv1 = np.zeros(np.shape(w_conv1));
	rw_conv2 = np.zeros(np.shape(w_conv2));
	rw_conv3 = np.zeros(np.shape(w_conv3));
	rw_conv4 = np.zeros(np.shape(w_conv4));
	rw_conv5 = np.zeros(np.shape(w_conv5));

	rw_conv1 = puf_layer.puf_layer(pw_conv1, pufs_conv1, ratio1, "conv1", "unreliable");
	rw_conv2 = puf_layer.puf_layer(pw_conv2, pufs_conv2, ratio2, "conv2", "unreliable");
	rw_conv3 = puf_layer.puf_layer(pw_conv3, pufs_conv3, ratio1, "conv3", "unreliable");
	rw_conv4 = puf_layer.puf_layer(pw_conv4, pufs_conv4, ratio1, "conv4", "unreliable");
	rw_conv5 = puf_layer.puf_layer(pw_conv5, pufs_conv5, ratio1, "conv5", "unreliable");

	print "recover unreliable end"

	net.params['conv1'][0].data[...] = rw_conv1;
	net.params['conv2'][0].data[...] = rw_conv2;
	net.params['conv3'][0].data[...] = rw_conv3;
	net.params['conv4'][0].data[...] = rw_conv4;
	net.params['conv5'][0].data[...] = rw_conv5;

	net.save(net_ori_recover_path);


	    #adjust
	print "adj"
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
		
	print "adjust end"

	net.params['conv1'][0].data[...] = aw_conv1;
	net.params['conv2'][0].data[...] = aw_conv2;
	net.params['conv3'][0].data[...] = aw_conv3;
	net.params['conv4'][0].data[...] = aw_conv4;
	net.params['conv5'][0].data[...] = aw_conv5;


	print "puf"
	pw_conv1 = puf_layer.puf_layer(aw_conv1, pufs_conv1, ratio1, "conv1", "reliable");
	pw_conv2 = puf_layer.puf_layer(aw_conv2, pufs_conv2, ratio2, "conv2", "reliable");
	pw_conv3 = puf_layer.puf_layer(aw_conv3, pufs_conv3, ratio1, "conv3",  "reliable");
	pw_conv4 = puf_layer.puf_layer(aw_conv4, pufs_conv4, ratio1, "conv4", "reliable");
	pw_conv5 = puf_layer.puf_layer(aw_conv5, pufs_conv5, ratio1, "conv5", "reliable");
	print "puf end"

	print "recover unreliable"
	rw_conv1 = puf_layer.puf_layer(pw_conv1, pufs_conv1, ratio1, "conv1", "unreliable");
	rw_conv2 = puf_layer.puf_layer(pw_conv2, pufs_conv2, ratio2, "conv2", "unreliable");
	rw_conv3 = puf_layer.puf_layer(pw_conv3, pufs_conv3, ratio1, "conv3", "unreliable");
	rw_conv4 = puf_layer.puf_layer(pw_conv4, pufs_conv4, ratio1, "conv4", "unreliable");
	rw_conv5 = puf_layer.puf_layer(pw_conv5, pufs_conv5, ratio1, "conv5", "unreliable");

	print "recover unreliable end"
	net.params['conv1'][0].data[...] = rw_conv1;
	net.params['conv2'][0].data[...] = rw_conv2;
	net.params['conv3'][0].data[...] = rw_conv3;
	net.params['conv4'][0].data[...] = rw_conv4;
	net.params['conv5'][0].data[...] = rw_conv5;

	net.save(net_adj_recover_path);



