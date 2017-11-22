import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import math

def main():
	
	tr_data = load_svmlight_file("train.txt");#To randomly select 5000 points

	XTR = np.matrix(tr_data[0].toarray()); # Converts sparse matrices to dense
	YTR = np.array(tr_data[1]).T; # The trainig labels

	ts_data = load_svmlight_file("test.txt");#To randomly select 5000 points

	XTS = np.matrix(ts_data[0].toarray()); # Converts sparse matrices to dense
	YTS = np.array(ts_data[1]).T; # The trainig labels

	W = np.zeros((15,100))
	B = np.zeros((15,100))
	Z = np.zeros((3,100))

	gamma = 0.5

	for i in range(XTR.shape[0]):
		x = XTR[i];
		y = YTR[i];

		for j in range(1,10):
			

		
	





if __name__ == '__main__':
	main()