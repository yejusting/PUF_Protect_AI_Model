import numpy as np
import random
import math

import float_to_binstr

def generate_int_challenge(bitlen, matkernel):
	k = len(matkernel);
	n = k * k;

	warray = matkernel.reshape(n);

	strarray = np.zeros(n);
	strarray = strarray.astype(np.str);
	for i in range (0, n):
		strarray[i] = float_to_binstr.float_to_binstr(warray[i]);

	ch = '';
	cnt = 0;
	enough = 0
	for i in range (0, 5):
		for j in range (0, n):
			s = strarray[j];
			l = len(s);
			ch = ch + s[l-i-1];
			cnt = cnt + 1;
			if(cnt == bitlen):
				enough = 1;
				break;
		if(enough):
			break;

	if(ch == '' ):
		ch = '0'


	challenge_int = int(ch, 2);

	return challenge_int;