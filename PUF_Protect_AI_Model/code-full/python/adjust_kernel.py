import numpy as np
import math
import random

import generate_int_challenge
import float_to_binstr
import adjustbinstr
import generate_reliability_from_str_kernel
import generate_floatkernel_from_strkernel



def adjust_kernel(matkernel, puf):
	size = len(puf);
	bitlen = int(math.log(size, 2));
	challenge_int = generate_int_challenge.generate_int_challenge(bitlen, matkernel);
	reliability = puf[challenge_int][1];
	if(reliability):
		return matkernel;
	else:
		k = len(matkernel);
		n = k * k;

		strarray = np.zeros((k, k));
		strarray = strarray.astype(np.str);
		for i in range (0, k):
			for j in range (0, k):
				strarray[i][j] = float_to_binstr.float_to_binstr(matkernel[i][j]);

		labels = np.zeros((k, k), dtype = np.int)
		cnt = 0;
		enough = 0
		for i in range (0, 5):
			for j in range (0, k):
				for l in range (0, k):
					cnt = cnt + 1;
					labels[j][l] = labels[j][l] + 1;
					if(cnt == bitlen):
						enough = 1;
						break;
				if(enough):
					break;
			if(enough):
				break;

		for i in range (0, k):
			for j in range (0, k):
				if(labels[i][j]):
					wstr = strarray[i][j];
					m_wstr = adjustbinstr.adjustbinstr(wstr, 1, 0);
					strarray[i][j] = m_wstr;
					reliability = generate_reliability_from_str_kernel.generate_reliability_from_str_kernel(strarray, puf);
					if(reliability):
						break;
					else:
						strarray[i][j] = m_wstr;
			if(reliability):
				break;
		kernel_out = generate_floatkernel_from_strkernel.generate_floatkernel_from_strkernel(strarray)

		if(reliability):
			return kernel_out;
		else:
			print "cannot adjust";
			return matkernel;