import numpy as np
#matplotlib is a library of presenting images
import matplotlib.pyplot as plt 
import pylab
import caffe
import os,sys

caffe_root='/home/gql/caffe/'
os.chdir(caffe_root)
sys.path.insert(0,caffe_root+'python')
im = caffe.io.load_image('examples/images/cat.jpg')
print im.shape
plt.imshow(im)
#follow imshow to show the image correctly
pylab.show()
plt.axis('off')