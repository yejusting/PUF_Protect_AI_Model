import math
import generate_int_challenge


def generate_response(matkernel, puf):
	bitlen = int(math.log(len(puf), 2));
	challenge_int = generate_int_challenge.generate_int_challenge(bitlen, matkernel);
	response = puf[challenge_int][0];
	return response
