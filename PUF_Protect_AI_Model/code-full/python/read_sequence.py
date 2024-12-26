import numpy as np
import random
def read_sequence(sequence, errorrate):
	shape = np.shape(sequence);
	r = np.zeros(shape);
	n = np.shape(r)[0];

	for i in range(0, n):
		r[i] = sequence[i];

	index = np.arange(n);

	errornum = 0;
	if(errorrate > 0):
		errornum = int(n * errorrate);
		error_indexes = random.sample(index, errornum);
		for j in range (0, errornum):
			error_index = error_indexes[j];
			r[error_index] = 1 - r[error_index];

	return r;