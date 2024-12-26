import numpy as np
import math

import generate_floatkernel_from_strkernel
import generate_int_challenge


def generate_reliability_from_str_kernel(strmatkernel, puf):
	k = len(strmatkernel);
	matkernel = generate_floatkernel_from_strkernel.generate_floatkernel_from_strkernel(strmatkernel);

	size = len(puf);
	bitlen = int(math.log(size, 2));
	challenge_int = generate_int_challenge.generate_int_challenge(bitlen, matkernel);
	reliability = puf[challenge_int][1];
	return reliability;