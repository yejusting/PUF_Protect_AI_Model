import numpy as np
import matplotlib.pyplot as plt
import pylab
import caffe
import sys, os, caffe


def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )

def show_data(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data,cmap='gray')
    plt.axis('off')
    plt.show()			#both plt.show() and pylab.show() are correct


if __name__=="__main__":

	caffe_root = '/home/gql/caffe/'
	sys.path.insert(0, caffe_root + 'python')
	os.chdir(caffe_root)

	if not os.path.isfile(caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel'):
		print("caffemodel does not exist...")

	caffe.set_mode_gpu()
	net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick.prototxt',
		     		caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel',
		     		caffe.TEST)

	binMean=caffe_root+'examples/cifar10/mean.binaryproto'
	npyMean=caffe_root+'examples/cifar10/mean.npy'
	convert_mean(binMean,npyMean)


	# preprocess data
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape}) 
	# channel first
	transformer.set_transpose('data',(2,0,1))  
	transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))
	# scale the pixel to [1-255]
	transformer.set_raw_scale('data',255)  
	#RGB-->BGR conversion
	transformer.set_channel_swap('data',(2,1,0)) 

	# load image
	im = caffe.io.load_image('examples/images/32.png')
	net.blobs['data'].data[...] = transformer.preprocess('data', im)
	inputData=net.blobs['data'].data

	# show the image before and after mean
	plt.figure()
	plt.subplot(1,2,1),plt.title("origin")
	plt.imshow(im)
	plt.axis('off')
	plt.subplot(1,2,2),plt.title("subtract mean")
	# depreprocess the image to show it
	plt.imshow(transformer.deprocess('data', inputData[0])) 
	plt.axis('off')
	plt.savefig('img.png') #to save the figure correctly, the savefig should be written before the show function
	plt.show() #show the two figures in one whiteboarc, after the two figures show, only need one


	net.forward()

	# print the data information in each layer, ip means Inner Product
	for k, v in net.blobs.items():
		print k, v.data.shape   

	#print the parameter information in each layer, v[0] means weights, v[1] means params, this function is the same to the function in the file read_model_parameter.py
	for k, v in net.params.items():
		print k, v[0].data.shape, v[1].data.shape 


	plt.rcParams['figure.figsize'] = (8, 8)
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

	# show_data(net.blobs['conv1'].data[0])
	# show_data(net.params['conv1'][0].data.reshape(32*3, 5, 5))
	# show_data(net.blobs['pool1'].data[0])
	# show_data(net.blobs['conv2'].data[0], padval=0.5)
	# show_data(net.params['conv2'][0].data.reshape(32**2,5,5))
	# show_data(net.blobs['conv3'].data[0],padval=0.5)
	# show_data(net.params['conv3'][0].data.reshape(64*32,5,5)[:1024])
	# show_data(net.blobs['pool3'].data[0],padval=0.2)
	# probability vector
	feat = net.blobs['prob'].data[0]
	print feat
	plt.plot(feat.flat)
	plt.show()

	









