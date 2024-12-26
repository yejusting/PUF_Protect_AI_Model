#cut the weight to 4 decimals
def baseweight(w):
	int_a = int(w * 10000)
	b = bin(int_a)   #convert int to binary str
	sequence = b.split("b");
	sign = sequence[0];
	value = sequence[1];
	validvalue = "";
	if(len(value) > 15):

		validvalue = value[0:15];
	else:
		validvalue = value;

	validbin = sign + "b" + validvalue;
	validint = int(validbin,2)
	m_f = float(validint) / 10000
	return m_f;

