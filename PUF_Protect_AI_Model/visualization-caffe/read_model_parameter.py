import numpy as np
import sys,os,caffe
import math
import random
import matplotlib.pyplot as plt
import pylab

if __name__=="__main__":
	caffe_root = '/home/gql/caffe/' 
	sys.path.insert(0, caffe_root + 'python')
	os.chdir(caffe_root)
	if not os.path.isfile(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'):
		print("caffemodel is not exist...")
	#use the trained model to set the test network
	caffe.set_mode_gpu()
	net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick.prototxt',
	                caffe_root + 'examples/cifar10/cifar10_quick_iter_4000 (copy).caffemodel',    
	               caffe.TEST)

	param_names = net.params.keys()  #layer names in the net, in a list

	for param_name in param_names:
		# weight parameter
		weight = net.params[param_name][0].data
		#bias parameter
		bias = net.params[param_name][1].data

		print np.shape(weight)
		print np.shape(bias)




