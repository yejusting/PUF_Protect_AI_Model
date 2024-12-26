import numpy as np
def generate_random01(num):
	cycle_num = 100;
	s = np.zeros(num, dtype = np.int);
	for i in range (0, cycle_num):
		s = np.random.randint(0, 2, size = num);
		if(np.sum(s==1) == num/2):
			break;
	return s;