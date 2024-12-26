import numpy as np
import random
import math

import adjust_kernel

def adjust_conv1_kernel(kernel, puf): #conv1kernel: 11 * 11
	pesize = 3;
	kernelsize = 11;
	arraysize = 3;
	kernelout = np.zeros((11, 11));

	kernel_out_array = np.zeros((9, 9));

	for i in range (0, arraysize):
		for j in range (0, arraysize):
			matkernel = kernel[3 * i : 3 * i + 3, 3 * j : 3 * j + 3];
			challenge_int = 100
			kernel_out_array[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = adjust_kernel.adjust_kernel(matkernel, puf);

	out1 = kernel[0:9, 9];
	out2 = kernel[0:9, 10];
	out3 = kernel[9, 0:9];
	out4 = kernel[10, 0:9];

	kernel_out1 = adjust_kernel.adjust_kernel(out1.reshape((3, 3)), puf)
	kernel_out2 = adjust_kernel.adjust_kernel(out2.reshape((3, 3)), puf)
	kernel_out3 = adjust_kernel.adjust_kernel(out3.reshape((3, 3)), puf)
	kernel_out4 = adjust_kernel.adjust_kernel(out4.reshape((3, 3)), puf)

	kernelout[0 : 9, 0 : 9] = kernel_out_array;

	kernelout[0 : 9, 9] = kernel_out1.reshape(9);
	kernelout[0 : 9, 10] = kernel_out2.reshape(9);
	kernelout[9, 0 : 9] = kernel_out3.reshape((1, 9));
	kernelout[10, 0 : 9] = kernel_out4.reshape((1, 9));

	kernelout[9 : 11, 9 : 11] = kernel[9 : 11, 9 : 11];

	return kernelout;