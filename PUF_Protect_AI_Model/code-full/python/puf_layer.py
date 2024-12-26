import numpy as np
import random
import math

import puf_conv1_kernel
import puf_kernel

def puf_layer(w, puf, ratio, layer, operation):
	w_shape = np.shape(w);
	puf_shape = np.shape(puf);

	out_batchsize = puf_shape[0];
	in_batchsize = puf_shape[1];

	w_puf = np.zeros(w_shape);
	if(layer == "conv1"):
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;
				cur_kernel = w[i][j];
				w_puf[i][j] = puf_conv1_kernel.puf_conv1_kernel(cur_kernel, puf[puf_row][puf_col], ratio, operation);
				
	else:
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;	
							
				cur_kernel = np.zeros(np.shape(w[i][j]))
				cur_kernel = w[i][j];
				w_puf[i][j] = puf_kernel.puf_kernel(cur_kernel, puf[puf_row][puf_col], ratio, operation);

				
	return w_puf;