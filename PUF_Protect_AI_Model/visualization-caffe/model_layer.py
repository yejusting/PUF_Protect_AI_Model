# http://www.cnblogs.com/denny402/p/5105911.html

#import nessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pylab
# matplotlib inline
import sys, os, caffe

#set current directory, test whether the model has been trained out
caffe_root = '/home/gql/caffe/'
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)

if not os.path.isfile(caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel'):
	print("caffemodel does not exist")

#set the tested model
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick.prototxt',
				caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel',
				caffe.TEST)


#convert the binary mean to python mean
def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )
binMean=caffe_root+'examples/cifar10/mean.binaryproto'
npyMean=caffe_root+'examples/cifar10/mean.npy'
convert_mean(binMean,npyMean)

#show data
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
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'







im = caffe.io.load_image('examples/images/32.png')

#load image into blog and sustract mean
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(npyMean).mean(1).mean(1)) # sustract mean
transformer.set_raw_scale('data', 255)  
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].data[...] = transformer.preprocess('data',im)
inputData=net.blobs['data'].data



#show the image before and after sustracting mean
plt.figure()
plt.subplot(1,2,1),plt.title("origin")
plt.imshow(im)
plt.axis('off')
pylab.show()
plt.subplot(1,2,2),plt.title("subtract mean")
plt.imshow(transformer.deprocess('data', inputData[0]))
plt.axis('off')
pylab.show()

net.forward()
for k, v in net.blobs.items():
	print (k, v.data.shape)

#show conv1 result
show_data(net.blobs['conv1'].data[0])
pylab.show()
print net.blobs['conv1'].data.shape
show_data(net.params['conv1'][0].data.reshape(32*3,5,5))
pylab.show()
print net.params['conv1'][0].data.shape


#show pool1 data
show_data(net.blobs['pool1'].data[0])
pylab.show()
net.blobs['pool1'].data.shape


#show conv2 data
show_data(net.blobs['conv2'].data[0],padval=0.5)
pylab.show()
print net.blobs['conv2'].data.shape
show_data(net.params['conv2'][0].data.reshape(32**2,5,5))
pylab.show()
print net.params['conv2'][0].data.shape



#show conv3 data
print net.blobs['conv3'].data.shape
print net.params['conv3'][0].data.shape
show_data(net.blobs['conv3'].data[0],padval=0.5)
pylab.show()
show_data(net.params['conv3'][0].data.reshape(64*32,5,5)[:1024])
pylab.show()


#show pool3 data
show_data(net.blobs['pool3'].data[0],padval=0.2)
pylab.show()
print net.blobs['pool3'].data.shape

#ip layer
feat = net.blobs['prob'].data[0]
print feat
plt.plot(feat.flat)
pylab.show()









