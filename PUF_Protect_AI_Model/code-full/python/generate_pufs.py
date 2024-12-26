import numpy as np
import random
import math

import generate_puf

def generate_pufs(lockrate, row, col, bitlength, errorrate):  #lockrate-protect rate of PEs;errorrate=1-reliability
	num_puf = row * col;
	pufsize = int(math.pow(2, bitlength));
	num_vpuf = int(math.ceil(float(num_puf * lockrate)));

	conv1_pufs = np.ones((num_puf, pufsize, 2), dtype = np.int); #[i][0]-response, [i][1] 

	if(num_puf == num_vpuf):
		for i in range (0, num_puf):
			conv1_pufs[i] = generate_puf.generate_puf(bitlength, errorrate);
	else:
		index_vpuf = random.sample(np.arange(num_puf), num_vpuf);

		for i in range (0, num_vpuf):
			index = index_vpuf[i];
			conv1_pufs[index] = generate_puf.generate_puf(bitlength, errorrate);

	pufs_formated = np.zeros((row, col, pufsize, 2), dtype = np.int);
	for i in range(0, row):
		for j in range (0, col):
			pufs_formated[i][j] = conv1_pufs[col * i + j];

	return pufs_formated;

