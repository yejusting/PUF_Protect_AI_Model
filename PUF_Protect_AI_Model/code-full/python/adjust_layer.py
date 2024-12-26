import numpy as np
import random
import math

import adjust_conv1_kernel
import adjust_kernel

def adjust_layer(w, puf, layer):
	w_shape = np.shape(w);
	puf_shape = np.shape(puf);

	out_batchsize = puf_shape[0];
	in_batchsize = puf_shape[1];

	w_adj = np.zeros(w_shape);
	a = 0;
	b = 0;
	if(layer == "conv1"):
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;
				cur_kernel = w[i][j];
				w_adj[i][j] = adjust_conv1_kernel.adjust_conv1_kernel(cur_kernel, puf[puf_row][puf_col]);
				
	else:
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;								
				cur_kernel = np.zeros(np.shape(w[i][j]))
				cur_kernel = w[i][j];
				w_adj[i][j] = adjust_kernel.adjust_kernel(cur_kernel, puf[puf_row][puf_col]);
				
				
	return w_adj;