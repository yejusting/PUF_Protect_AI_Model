import numpy as np

import binstr_to_float

def generate_floatkernel_from_strkernel(strkernel):
	k = len(strkernel);
	matkernel = np.zeros((k, k));
	for i in range (0, k):
		for j in range (0, k):
			binstr = strkernel[i][j];
			w = binstr_to_float.binstr_to_float(binstr);
			matkernel[i][j] = w;

	return matkernel