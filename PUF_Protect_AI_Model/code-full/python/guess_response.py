import numpy as np
def guess_response(puf):
	size = len(puf);
	base = np.ones((size, 2), dtype = np.int)
	response = 1;

	if((puf == base).all()):
		response = 1;
	else:
		response = np.random.randint(0, 2);

	return response