import generate_pufs
def pufs_preparation(lockrate, bitlength, errorrate):
	pufs_conv1 = generate_pufs.generate_pufs(lockrate, 16, 3, bitlength, errorrate);
	pufs_conv2 = generate_pufs.generate_pufs(lockrate, 2, 16, bitlength, errorrate);
	pufs_conv3 = generate_pufs.generate_pufs(lockrate, 3, 16, bitlength, errorrate);
	pufs_conv4 = generate_pufs.generate_pufs(lockrate, 3, 12, bitlength, errorrate);
	pufs_conv5 = generate_pufs.generate_pufs(lockrate, 2, 16, bitlength, errorrate);

	np.save("/home/gql/Desktop/debug/pufs_conv1.npy", pufs_conv1);
	np.save("/home/gql/Desktop/debug/pufs_conv2.npy", pufs_conv2);
	np.save("/home/gql/Desktop/debug/pufs_conv3.npy", pufs_conv3);
	np.save("/home/gql/Desktop/debug/pufs_conv4.npy", pufs_conv4);
	np.save("/home/gql/Desktop/debug/pufs_conv5.npy", pufs_conv5);