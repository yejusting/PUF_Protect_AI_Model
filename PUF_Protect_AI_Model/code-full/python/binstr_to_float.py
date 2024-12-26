def binstr_to_float(binstr):
	w_int = int(binstr, 2);
	f = float(w_int) / 10000;
	return f;