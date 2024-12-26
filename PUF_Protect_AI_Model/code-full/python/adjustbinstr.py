def adjustbinstr(binstr, end1, end2): #if end1 is 1, the least significant bit is reversed; if end2 is 1 the least significant bit is reversed
	lst = list(binstr);
	lstsize = len(lst);

	if(end1): 
		if lst[lstsize - 1] == '1':
			lst[lstsize - 1] = '0'
		else:
			lst[lstsize - 1] = '1'

	if(end2):
		if lst[lstsize - 2] == '1':
			lst[lstsize - 2] = '0'
		else:
			lst[lstsize - 2] = '1'

	s = ''.join(lst) 
	return s;