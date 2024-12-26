import numpy as np
import math

import generate_response
import generate_unreliable_response
import guess_response
def puf_kernel(matkernel, puf, ratio, operation):
	kernel_out = np.zeros(np.shape(matkernel));
	# kernel_out = matkernel
	k = np.shape(kernel_out)[0];#kernel_size
	n = math.floor(k * k * ratio);

	l = 0;
	s = 0;

	# response = generate_unreliable_response(matkernel, puf)
	if(operation == "reliable"):		
		response = generate_response.generate_response(matkernel, puf)
	elif(operation == "unreliable"):
		response = generate_unreliable_response.generate_unreliable_response(matkernel, puf);
	elif(operation == "guess"):
		response = guess_response.guess_response(puf)
	else:
		print "no such operation"


	if(not response):		
		for i in range(0, k):
			for j in range(0, k):
				l = l + 1;
				if(l <= n):
					kernel_out[i][j] = -matkernel[i][j];
				else:
					kernel_out[i][j] = matkernel[i][j];
					

	else:
		for i in range (0, k):
			for j in range (0, k):
				kernel_out[i][j] = matkernel[i][j];

	return kernel_out;
