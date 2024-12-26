import math
import generate_int_challenge

def generate_unreliable_response(matkernel, puf):
	bitlen = int(math.log(len(puf), 2))
	challenge_int = generate_int_challenge.generate_int_challenge(bitlen, matkernel)

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