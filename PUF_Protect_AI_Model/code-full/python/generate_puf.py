import numpy as np
import math

import generate_random01
import read_sequence


def generate_puf(bitlength, errorrate):
	puf_size = int(math.pow(2, bitlength));
	responses = generate_random01.generate_random01(puf_size);
	reliabilities = read_sequence.read_sequence(np.ones(puf_size, dtype = np.int), errorrate);

	puf = np.zeros((puf_size, 2), dtype = np.int);
	for i in range (0, puf_size):
		puf[i][0] = responses[i];
		puf[i][1] = int(reliabilities[i]); #1-reliable;0-unreliable;

	puf[0][0] = 1;

	return puf;