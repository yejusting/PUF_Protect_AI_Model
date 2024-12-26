import numpy as np
import random
import math
import sys,os,caffe


caffe_root = '/home/gql/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)


#use the trained model to set the test network
caffe.set_mode_gpu()

vec_neg = np.vectorize(lambda x: x + math.pow(10, -18));


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

def binstr_to_float(binstr):
	w_int = int(binstr, 2);
	f = float(w_int) / 10000;
	return f;


def generate_random01(num):
	cycle_num = 100;
	s = np.zeros(num, dtype = np.int);
	for i in range (0, cycle_num):
		s = np.random.randint(0, 2, size = num);
		if(np.sum(s==1) == num/2):
			break;
	return s;


def generate_puf(bitlength, errorrate):
	puf_size = int(math.pow(2, bitlength));
	responses = generate_random01(puf_size);
	reliabilities = read_sequence(np.ones(puf_size, dtype = np.int), errorrate);

	puf = np.zeros((puf_size, 2), dtype = np.int);
	for i in range (0, puf_size):
		puf[i][0] = responses[i];
		puf[i][1] = int(reliabilities[i]); #1-reliable;0-unreliable;

	puf[0][0] = 1;

	return puf;

def generate_pufs(lockrate, row, col, bitlength, errorrate):  #lockrate-protect rate of PEs;errorrate=1-reliability
	num_puf = row * col;
	pufsize = int(math.pow(2, bitlength));
	num_vpuf = int(math.ceil(float(num_puf * lockrate)));

	conv1_pufs = np.ones((num_puf, pufsize, 2), dtype = np.int); #[i][0]-response, [i][1] 

	if(num_puf == num_vpuf):
		for i in range (0, num_puf):
			conv1_pufs[i] = generate_puf(bitlength, errorrate);
	else:
		index_vpuf = random.sample(np.arange(num_puf), num_vpuf);

		for i in range (0, num_vpuf):
			index = index_vpuf[i];
			conv1_pufs[index] = generate_puf(bitlength, errorrate);

	pufs_formated = np.zeros((row, col, pufsize, 2), dtype = np.int);
	for i in range(0, row):
		for j in range (0, col):
			pufs_formated[i][j] = conv1_pufs[col * i + j];

	return pufs_formated;

def generate_int_challenge(bitlen, matkernel):
	k = len(matkernel);
	n = k * k;

	warray = matkernel.reshape(n);

	strarray = np.zeros(n);
	strarray = strarray.astype(np.str);
	for i in range (0, n):
		strarray[i] = float_to_binstr(warray[i]);

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



def generate_response(matkernel, puf):
	bitlen = int(math.log(len(puf), 2));
	challenge_int = generate_int_challenge(bitlen, matkernel);
	response = puf[challenge_int][0];
	return response

def generate_unreliable_response(matkernel, puf):
	bitlen = int(math.log(len(puf), 2))
	challenge_int = generate_int_challenge(bitlen, matkernel)

	relresponse = puf[challenge_int][0]
	reliable = puf[challenge_int][1]
	if(not reliable):
		print "not reliable"

	response = 0;
	if(reliable):
		response = relresponse;
	else:
		response = 1 - relresponse;

	return response

def guess_response(puf):
	size = len(puf);
	base = np.ones((size, 2), dtype = np.int)
	response = 1;

	if((puf == base).all()):
		response = 1;
	else:
		response = np.random.randint(0, 2);

	return response



def puf_kernel(matkernel, puf, ratio, operation):
	kernel_out = np.zeros(np.shape(matkernel));
	# kernel_out = matkernel
	k = np.shape(kernel_out)[0];#kernel_size
	n = math.floor(k * k * ratio);

	l = 0;
	s = 0;

	# response = generate_unreliable_response(matkernel, puf)
	if(operation == "reliable"):		
		response = generate_response(matkernel, puf)
	elif(operation == "unreliable"):
		response = generate_unreliable_response(matkernel, puf);
	elif(operation == "guess"):
		response = guess_response(puf)
	else:
		print "no such operation"


	if(not response):		
		for i in range(0, k):
			for j in range(0, k):
				l = l + 1;
				if(l <= n):
					kernel_out[i][j] = -matkernel[i][j];
				else:
					kernel_out[i][j] = matkernel[i][j];
					

	else:
		for i in range (0, k):
			for j in range (0, k):
				kernel_out[i][j] = matkernel[i][j];

	return kernel_out; 

def generate_reliability_from_str_kernel(strmatkernel, puf):
	k = len(strmatkernel);
	matkernel = generate_floatkernel_from_strkernel(strmatkernel);

	size = len(puf);
	bitlen = int(math.log(size, 2));
	challenge_int = generate_int_challenge(bitlen, matkernel);
	reliability = puf[challenge_int][1];
	return reliability;

def generate_floatkernel_from_strkernel(strkernel):
	k = len(strkernel);
	matkernel = np.zeros((k, k));
	for i in range (0, k):
		for j in range (0, k):
			binstr = strkernel[i][j];
			w = binstr_to_float(binstr);
			matkernel[i][j] = w;

	return matkernel



def adjust_kernel(matkernel, puf):
	size = len(puf);
	bitlen = int(math.log(size, 2));
	challenge_int = generate_int_challenge(bitlen, matkernel);
	reliability = puf[challenge_int][1];
	if(reliability):
		return matkernel;
	else:
		k = len(matkernel);
		n = k * k;

		strarray = np.zeros((k, k));
		strarray = strarray.astype(np.str);
		for i in range (0, k):
			for j in range (0, k):
				strarray[i][j] = float_to_binstr(matkernel[i][j]);

		labels = np.zeros((k, k), dtype = np.int)
		cnt = 0;
		enough = 0
		for i in range (0, 5):
			for j in range (0, k):
				for l in range (0, k):
					cnt = cnt + 1;
					labels[j][l] = labels[j][l] + 1;
					if(cnt == bitlen):
						enough = 1;
						break;
				if(enough):
					break;
			if(enough):
				break;

		for i in range (0, k):
			for j in range (0, k):
				if(labels[i][j]):
					wstr = strarray[i][j];
					m_wstr = adjustbinstr(wstr, 1, 0);
					strarray[i][j] = m_wstr;
					reliability = generate_reliability_from_str_kernel(strarray, puf);
					if(reliability):
						break;
					else:
						strarray[i][j] = m_wstr;
			if(reliability):
				break;
		kernel_out = generate_floatkernel_from_strkernel(strarray)

		if(reliability):
			return kernel_out;
		else:
			print "cannot adjust";
			return matkernel;


def puf_conv1_kernel(kernel, puf, ratio, operation): #conv1kernel: 11 * 11
	pesize = 3;
	kernelsize = 11;
	arraysize = 3;
	kernelout = np.zeros((11, 11));

	kernel_out_array = np.zeros((9, 9));

	for i in range (0, arraysize):
		for j in range (0, arraysize):
			matkernel = kernel[3 * i : 3 * i + 3, 3 * j : 3 * j + 3];
			kernel_out_array[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = puf_kernel(matkernel, puf, ratio, operation);

	out1 = kernel[0:9, 9];
	out2 = kernel[0:9, 10];
	out3 = kernel[9, 0:9];
	out4 = kernel[10, 0:9];

	kernel_out1 = puf_kernel(out1.reshape((3, 3)), puf, ratio, operation)
	kernel_out2 = puf_kernel(out2.reshape((3, 3)), puf, ratio, operation)
	kernel_out3 = puf_kernel(out3.reshape((3, 3)), puf, ratio, operation)
	kernel_out4 = puf_kernel(out4.reshape((3, 3)), puf, ratio, operation)

	kernelout[0 : 9, 0 : 9] = kernel_out_array;
	kernelout[0 : 9, 9] = kernel_out1.reshape(9);
	kernelout[0 : 9, 10] = kernel_out2.reshape(9);
	kernelout[9, 0 : 9] = kernel_out3.reshape((1, 9));
	kernelout[10, 0 : 9] = kernel_out4.reshape((1, 9));

	kernelout[9 : 11, 9 : 11] = kernel[9 : 11, 9 : 11];


	return kernelout;



def adjust_conv1_kernel(kernel, puf): #conv1kernel: 11 * 11
	pesize = 3;
	kernelsize = 11;
	arraysize = 3;
	kernelout = np.zeros((11, 11));

	kernel_out_array = np.zeros((9, 9));

	for i in range (0, arraysize):
		for j in range (0, arraysize):
			matkernel = kernel[3 * i : 3 * i + 3, 3 * j : 3 * j + 3];
			challenge_int = 100
			kernel_out_array[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = adjust_kernel(matkernel, puf);

	out1 = kernel[0:9, 9];
	out2 = kernel[0:9, 10];
	out3 = kernel[9, 0:9];
	out4 = kernel[10, 0:9];

	kernel_out1 = adjust_kernel(out1.reshape((3, 3)), puf)
	kernel_out2 = adjust_kernel(out2.reshape((3, 3)), puf)
	kernel_out3 = adjust_kernel(out3.reshape((3, 3)), puf)
	kernel_out4 = adjust_kernel(out4.reshape((3, 3)), puf)

	kernelout[0 : 9, 0 : 9] = kernel_out_array;

	kernelout[0 : 9, 9] = kernel_out1.reshape(9);
	kernelout[0 : 9, 10] = kernel_out2.reshape(9);
	kernelout[9, 0 : 9] = kernel_out3.reshape((1, 9));
	kernelout[10, 0 : 9] = kernel_out4.reshape((1, 9));

	kernelout[9 : 11, 9 : 11] = kernel[9 : 11, 9 : 11];

	return kernelout;


def puf_layer(w, puf, ratio, layer, operation):
	w_shape = np.shape(w);
	puf_shape = np.shape(puf);

	out_batchsize = puf_shape[0];
	in_batchsize = puf_shape[1];

	w_puf = np.zeros(w_shape);
	if(layer == "conv1"):
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;
				cur_kernel = w[i][j];
				w_puf[i][j] = puf_conv1_kernel(cur_kernel, puf[puf_row][puf_col], ratio, operation);
				
	else:
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;	
							
				cur_kernel = np.zeros(np.shape(w[i][j]))
				cur_kernel = w[i][j];
				w_puf[i][j] = puf_kernel(cur_kernel, puf[puf_row][puf_col], ratio, operation);

				
	return w_puf;

def pufunlayer(w, puf, ratio, layer):
	w_shape = np.shape(w);
	puf_shape = np.shape(puf);

	out_batchsize = puf_shape[0];
	in_batchsize = puf_shape[1];

	w_puf = np.zeros(w_shape);
	if(layer == "conv1"):
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;
				cur_kernel = w[i][j];
				w_puf[i][j] = puf_unreliable_conv1_kernel(cur_kernel, puf[puf_row][puf_col], ratio);
				
	else:
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;	
							
				cur_kernel = np.zeros(np.shape(w[i][j]))
				cur_kernel = w[i][j];
				w_puf[i][j] = puf_unreliable_kernel(cur_kernel, puf[puf_row][puf_col], ratio);

				
	return w_puf;

def puf_unreliable_conv1_kernel(kernel, puf, ratio): #conv1kernel: 11 * 11
	pesize = 3;
	kernelsize = 11;
	arraysize = 3;
	kernelout = np.zeros((11, 11));

	kernel_out_array = np.zeros((9, 9));

	for i in range (0, arraysize):
		for j in range (0, arraysize):
			matkernel = kernel[3 * i : 3 * i + 3, 3 * j : 3 * j + 3];
			kernel_out_array[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = puf_unreliable_kernel(matkernel, puf, ratio);

	out1 = kernel[0:9, 9];
	out2 = kernel[0:9, 10];
	out3 = kernel[9, 0:9];
	out4 = kernel[10, 0:9];

	kernel_out1 = puf_unreliable_kernel(out1.reshape((3, 3)), puf, ratio)
	kernel_out2 = puf_unreliable_kernel(out2.reshape((3, 3)), puf, ratio)
	kernel_out3 = puf_unreliable_kernel(out3.reshape((3, 3)), puf, ratio)
	kernel_out4 = puf_unreliable_kernel(out4.reshape((3, 3)), puf, ratio)

	kernelout[0 : 9, 0 : 9] = kernel_out_array;
	kernelout[0 : 9, 9] = kernel_out1.reshape(9);
	kernelout[0 : 9, 10] = kernel_out2.reshape(9);
	kernelout[9, 0 : 9] = kernel_out3.reshape((1, 9));
	kernelout[10, 0 : 9] = kernel_out4.reshape((1, 9));

	kernelout[9 : 11, 9 : 11] = kernel[9 : 11, 9 : 11];


	return kernelout;		

def puf_unreliable_kernel(matkernel, puf, ratio):
	kernel_out = np.zeros(np.shape(matkernel));
	# kernel_out = matkernel
	k = np.shape(kernel_out)[0];#kernel_size
	n = math.ceil(k * k * ratio);

	l = 0;
	s = 0;

	response = generate_unreliable_response(matkernel, puf)
	# response = generate_response(matkernel, puf)

	if(not response):		
		for i in range(0, k):
			for j in range(0, k):
				l = l + 1;
				if(l <= n):
					kernel_out[i][j] = -matkernel[i][j];
				else:
					kernel_out[i][j] = matkernel[i][j];
					

	else:
		for i in range (0, k):
			for j in range (0, k):
				kernel_out[i][j] = matkernel[i][j];

	return kernel_out;


def adjust_layer(w, puf, layer):
	w_shape = np.shape(w);
	puf_shape = np.shape(puf);

	out_batchsize = puf_shape[0];
	in_batchsize = puf_shape[1];

	w_adj = np.zeros(w_shape);
	a = 0;
	b = 0;
	if(layer == "conv1"):
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;
				cur_kernel = w[i][j];
				w_adj[i][j] = adjust_conv1_kernel(cur_kernel, puf[puf_row][puf_col]);
				
	else:
		for i in range (0, w_shape[0]):
			for j in range (0, w_shape[1]):
				puf_row = i % out_batchsize;
				puf_col = j % in_batchsize;								
				cur_kernel = np.zeros(np.shape(w[i][j]))
				cur_kernel = w[i][j];
				w_adj[i][j] = adjust_kernel(cur_kernel, puf[puf_row][puf_col]);

				
	return w_adj;

def pufs_preparation(lockrate, bitlength, errorrate):
	pufs_conv1 = generate_pufs(lockrate, 16, 3, bitlength, errorrate);
	pufs_conv2 = generate_pufs(lockrate, 2, 16, bitlength, errorrate);
	pufs_conv3 = generate_pufs(lockrate, 3, 16, bitlength, errorrate);
	pufs_conv4 = generate_pufs(lockrate, 3, 12, bitlength, errorrate);
	pufs_conv5 = generate_pufs(lockrate, 2, 16, bitlength, errorrate);

	np.save("/home/gql/Desktop/debug/pufs_conv1.npy", pufs_conv1);
	np.save("/home/gql/Desktop/debug/pufs_conv2.npy", pufs_conv2);
	np.save("/home/gql/Desktop/debug/pufs_conv3.npy", pufs_conv3);
	np.save("/home/gql/Desktop/debug/pufs_conv4.npy", pufs_conv4);
	np.save("/home/gql/Desktop/debug/pufs_conv5.npy", pufs_conv5);




def puf_alexnet(netinpath, netoutpath, pufs_conv1, pufs_conv2, pufs_conv3, pufs_conv4, pufs_conv5, ratio1, ratio2, operation):
	print "path"
	print ("netinpath", netinpath);
	print ("netoutpath", netoutpath);

	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
	               caffe_root + netinpath,    
	              caffe.TEST);
	print "path"
	print ("netinpath", netinpath);
	print ("netoutpath", netoutpath);

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data


	pw_conv1 = np.zeros(np.shape(w_conv1));
	pw_conv2 = np.zeros(np.shape(w_conv2));
	pw_conv3 = np.zeros(np.shape(w_conv3));
	pw_conv4 = np.zeros(np.shape(w_conv4));
	pw_conv5 = np.zeros(np.shape(w_conv5));

	pw_conv1 = puf_layer(w_conv1, pufs_conv1, ratio1, "conv1", operation);
	pw_conv2 = puf_layer(w_conv2, pufs_conv2, ratio2, "conv2", operation);
	pw_conv3 = puf_layer(w_conv3, pufs_conv3, ratio1, "conv3", operation);
	pw_conv4 = puf_layer(w_conv4, pufs_conv4, ratio1, "conv4", operation);
	pw_conv5 = puf_layer(w_conv5, pufs_conv5, ratio1, "conv5", operation);

	net.params['conv1'][0].data[...] = pw_conv1;
	net.params['conv2'][0].data[...] = pw_conv2;
	net.params['conv3'][0].data[...] = pw_conv3;
	net.params['conv4'][0].data[...] = pw_conv4;
	net.params['conv5'][0].data[...] = pw_conv5;

	net.save(netoutpath);

def adjust_alexnet(netinpath, netoutpath):
	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
	               caffe_root + netinpath,    
	              caffe.TEST);

	pufs_conv1 = np.load("/home/gql/Desktop/debug/pufs_conv1.npy");
	pufs_conv2 = np.load("/home/gql/Desktop/debug/pufs_conv2.npy");
	pufs_conv3 = np.load("/home/gql/Desktop/debug/pufs_conv3.npy");
	pufs_conv4 = np.load("/home/gql/Desktop/debug/pufs_conv4.npy");
	pufs_conv5 = np.load("/home/gql/Desktop/debug/pufs_conv5.npy");

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data


	aw_conv1 = np.zeros(np.shape(w_conv1));
	aw_conv2 = np.zeros(np.shape(w_conv2));
	aw_conv3 = np.zeros(np.shape(w_conv3));
	aw_conv4 = np.zeros(np.shape(w_conv4));
	aw_conv5 = np.zeros(np.shape(w_conv5));


	aw_conv1 = adjust_layer(w_conv1, pufs_conv1, "conv1");
	aw_conv2 = adjust_layer(w_conv2, pufs_conv2, "conv2");
	aw_conv3 = adjust_layer(w_conv3, pufs_conv3, "conv3");
	aw_conv4 = adjust_layer(w_conv4, pufs_conv4, "conv4");
	aw_conv5 = adjust_layer(w_conv5, pufs_conv5, "conv5");


	net.params['conv1'][0].data[...] = aw_conv1;
	net.params['conv2'][0].data[...] = aw_conv2;
	net.params['conv3'][0].data[...] = aw_conv3;
	net.params['conv4'][0].data[...] = aw_conv4;
	net.params['conv5'][0].data[...] = aw_conv5;

   

	net.save(netoutpath);


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


def base_alexnet(netinpath, netoutpath):
	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
           caffe_root + netinpath,    
          caffe.TEST);

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data


	shape1 = np.shape(w_conv1);
	shape2 = np.shape(w_conv2);
	shape3 = np.shape(w_conv3);
	shape4 = np.shape(w_conv4);
	shape5 = np.shape(w_conv5);

	for i in range (0, shape1[0]):
		for j in range (0, shape1[1]):
			for k in range (0, shape1[2]):
				for l in range (0, shape1[3]):
					w_conv1[i][j][k][l] = baseweight(w_conv1[i][j][k][l]);

	for i in range (0, shape2[0]):
		for j in range (0, shape2[1]):
			for k in range (0, shape2[2]):
				for l in range (0, shape2[3]):
					w_conv2[i][j][k][l] = baseweight(w_conv2[i][j][k][l]);

	for i in range (0, shape3[0]):
		for j in range (0, shape3[1]):
			for k in range (0, shape3[2]):
				for l in range (0, shape3[3]):
					w_conv3[i][j][k][l] = baseweight(w_conv3[i][j][k][l]);

	for i in range (0, shape4[0]):
		for j in range (0, shape4[1]):
			for k in range (0, shape4[2]):
				for l in range (0, shape4[3]):
					w_conv4[i][j][k][l] = baseweight(w_conv4[i][j][k][l]);
	for i in range (0, shape5[0]):
		for j in range (0, shape5[1]):
			for k in range (0, shape5[2]):
				for l in range (0, shape5[3]):
					w_conv5[i][j][k][l] = baseweight(w_conv5[i][j][k][l]);



	net.params['conv1'][0].data[...] = w_conv1;
	net.params['conv2'][0].data[...] = w_conv2;
	net.params['conv3'][0].data[...] = w_conv3;
	net.params['conv4'][0].data[...] = w_conv4;
	net.params['conv5'][0].data[...] = w_conv5;

	net.save(netoutpath);	



if __name__=="__main__":
    #parameters
	netoutpath = 'models/bvlc_alexnet/pufmodel/reliability/recover1.caffemodel'
	netinpath = 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

	bitlength = 12
	errorrate = 0.08
	ratio1 = 0.9;
	ratio2 = 0.96;

	lockrate = 0.8
	pufs_conv1 = generate_pufs(lockrate, 16, 3, bitlength, errorrate);
	pufs_conv2 = generate_pufs(lockrate, 2, 16, bitlength, errorrate);
	pufs_conv3 = generate_pufs(lockrate, 3, 16, bitlength, errorrate);
	pufs_conv4 = generate_pufs(lockrate, 3, 12, bitlength, errorrate);
	pufs_conv5 = generate_pufs(lockrate, 2, 16, bitlength, errorrate);

	net = caffe.Net(caffe_root + 'models/bvlc_alexnet/train_val.prototxt',
               caffe_root + netinpath,    
              caffe.TEST);

	w_conv1 = net.params['conv1'][0].data
	w_conv2 = net.params['conv2'][0].data
	w_conv3 = net.params['conv3'][0].data
	w_conv4 = net.params['conv4'][0].data
	w_conv5 = net.params['conv5'][0].data

	print "puf"
	pw_conv1 = np.zeros(np.shape(w_conv1));
	pw_conv2 = np.zeros(np.shape(w_conv2));
	pw_conv3 = np.zeros(np.shape(w_conv3));
	pw_conv4 = np.zeros(np.shape(w_conv4));
	pw_conv5 = np.zeros(np.shape(w_conv5));

	pw_conv1 = puf_layer(w_conv1, pufs_conv1, ratio1, "conv1", "reliable");
	pw_conv2 = puf_layer(w_conv2, pufs_conv2, ratio2, "conv2", "reliable");
	pw_conv3 = puf_layer(w_conv3, pufs_conv3, ratio1, "conv3",  "reliable");
	pw_conv4 = puf_layer(w_conv4, pufs_conv4, ratio1, "conv4", "reliable");
	pw_conv5 = puf_layer(w_conv5, pufs_conv5, ratio1, "conv5", "reliable");
	print "puf end"

	print "recover unreliable"
	rw_conv1 = np.zeros(np.shape(w_conv1));
	rw_conv2 = np.zeros(np.shape(w_conv2));
	rw_conv3 = np.zeros(np.shape(w_conv3));
	rw_conv4 = np.zeros(np.shape(w_conv4));
	rw_conv5 = np.zeros(np.shape(w_conv5));

	rw_conv1 = puf_layer(pw_conv1, pufs_conv1, ratio1, "conv1", "unreliable");
	rw_conv2 = puf_layer(pw_conv2, pufs_conv2, ratio2, "conv2", "unreliable");
	rw_conv3 = puf_layer(pw_conv3, pufs_conv3, ratio1, "conv3", "unreliable");
	rw_conv4 = puf_layer(pw_conv4, pufs_conv4, ratio1, "conv4", "unreliable");
	rw_conv5 = puf_layer(pw_conv5, pufs_conv5, ratio1, "conv5", "unreliable");
	print "recover unreliable end"

	net.params['conv1'][0].data[...] = rw_conv1;
	net.params['conv2'][0].data[...] = rw_conv2;
	net.params['conv3'][0].data[...] = rw_conv3;
	net.params['conv4'][0].data[...] = rw_conv4;
	net.params['conv5'][0].data[...] = rw_conv5;

	net.save(netoutpath);



	# netoutfolder = outfolder + 'group' + str(lockrate) + '/';
	# netguesspath = '';
	# iteNum = 100;
	# for index in range(0, iteNum):
	# 	print index
	# 	pw_conv1 = np.zeros(np.shape(w_conv1));
	# 	pw_conv2 = np.zeros(np.shape(w_conv2));
	# 	pw_conv3 = np.zeros(np.shape(w_conv3));
	# 	pw_conv4 = np.zeros(np.shape(w_conv4));
	# 	pw_conv5 = np.zeros(np.shape(w_conv5));

	# 	gw_conv1 = puf_layer(w_conv1, pufs_conv1, ratio1, "conv1", "guess");
	# 	gw_conv2 = puf_layer(w_conv2, pufs_conv2, ratio2, "conv2", "guess");
	# 	gw_conv3 = puf_layer(w_conv3, pufs_conv3, ratio1, "conv3", "guess");
	# 	gw_conv4 = puf_layer(w_conv4, pufs_conv4, ratio1, "conv4", "guess");
	# 	gw_conv5 = puf_layer(w_conv5, pufs_conv5, ratio1, "conv5", "guess");

	# 	net.params['conv1'][0].data[...] = gw_conv1;
	# 	net.params['conv2'][0].data[...] = gw_conv2;
	# 	net.params['conv3'][0].data[...] = gw_conv3;
	# 	net.params['conv4'][0].data[...] = gw_conv4;
	# 	net.params['conv5'][0].data[...] = gw_conv5;

	# 	netguesspath = netoutfolder + "guess" + str(index) + '.caffemodel';
	# 	net.save(netguesspath);
