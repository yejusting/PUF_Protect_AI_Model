def float_to_binstr(w):
	int_w = int(w * 10000);
	bin_w = bin(int_w);
	sequence = bin_w.split('b');
	sign = sequence[0];
	binstr = sequence[1];

	while(len(binstr) < 15):
		binstr = '0' + binstr
	binstr = sign + 'b' + binstr

	return binstr