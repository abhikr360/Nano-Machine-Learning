import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import math

def z_gradient(x, y, Y, w, B, Z, gamma):
	# calculating A
	
	wx = w.dot(x.T)
	diff = B-wx
	###print(B.shape)
	###print(wx.shape)
	###print(diff.shape)
	diff = np.square(diff)
	###print(diff.shape)
	diff = np.sum(diff, axis=0)
	###print(diff.shape)
	diff = -1*gamma*gamma*diff
	###print(diff.shape)
	kernel = np.exp(diff)
	###print(kernel.shape)
	###print(Y.shape)
	A = -2*np.multiply(Y,kernel);
	###print(A.shape)
	###print(A)
	#calculating X
	# ### ###print(Z.shape)
	# ### ###print
	col = Z.dot(kernel.T)
	# ### ###print(col.shape)
	X = 2*col.dot(kernel)
	###print(X)
	return A+X


def test_z_gradient(x, y, Y, w, B, Z, gamma):
	wx = w.dot(x.T)
	diff = B-wx
	diff = np.square(diff)
	diff = np.sum(diff, axis=0)
	### ###print(diff.shape)
	#quit()
	diff = -1*gamma*gamma*(diff.T)
	kernel = np.exp(diff)

	X = np.array(B)
	B = np.array(B)
	kernel = np.array(kernel)
	### ###print(kernel.shape)

	### ###print(B.shape)

	for i in range (B.shape[0]):
		for j in range (B.shape[1]):
			temp = 4*(Y.T)
			#### ###print(Y.T)
			#### ###print(temp)
			temp = temp.dot(Z.T[j].T)
			temp = temp*gamma*gamma
			#### ###print(kernel.shape,i,j)
			temp = temp*kernel[j]
			X[i][j] = temp*(B[i][j] - np.array(wx[i]))
	### ###print(Y)
	### ###print(y)
	### ###print(X)



def b_gradient(x, y, Y, w, B, Z, gamma, org_dim, num_prot, proj_dim):
	wx = w.dot(x.T)
	diff = B-wx

	### ###print("--------------")

	kernel = np.square(diff)
	kernel = np.sum(kernel, axis=0)
	kernel = -1*gamma*gamma*kernel
	kernel = np.exp(kernel)
	# ### ###print(kernel.shape)
	yz = ((Y.T).dot(Z)).T
	yzK = np.multiply(yz, kernel.T)
	# ### ###print(yzK.shape)
	# ### ###print(diff.T.shape)
	X = 4*gamma*gamma*np.multiply(yzK, diff.T)
	X=X.T
	# ### ###print(X.shape)

	# X done
	zK = np.multiply(kernel.T, Z.T).T
	# ### ###print(zK.shape)
	zKsum = np.sum(zK, axis=1)
	# ### ###print(zKsum.shape)
	# zKsum = np.repeat(zKsum,num_prot,axis=1)
	# ### ###print(zKsum.shape)
	
	zKnew = np.multiply(Z.T, kernel.T).T
	# ### ###print(zKnew.shape)

	ZZ = np.dot(zKsum.T, zKnew)
	# ### ###print(ZZ.shape)

	# ZZ = np.repeat(ZZ,proj_dim,axis=0)
	fin = (-4*gamma*gamma*np.multiply(ZZ.T, diff.T)).T
	# zKsumK = np.multiply(kernel, zKsum)
	# ### ###print(zKsumK.shape)
	# dot = (Z.T).dot(zKsumK)
	

	# ### ###print(dot.shape)
	# diffdot = np.multiply((diff.T), dot)
	# fin = -4*gamma*gamma*diffdot
	return X+fin


def kernelm(v1, v2, gamma):
	gamma_sq = gamma * gamma
	# # ###print("gamma_sq")
	# # ###print(gamma_sq)
	# ###print((((v1-v2).T).dot(v1-v2)))
	return np.exp( -gamma_sq * (((v1-v2).T).dot(v1-v2)) )

def kernel(x, y, gamma):
	diff = x-y
	norm = np.linalg.norm(diff)
	k= np.exp(-1*gamma*gamma*norm*norm)
	### ###print(k.shape)
	return k

def testz_gradient(x, Y, W, B, Z, gamma):
	L = Z.shape[0]
	m = Z.shape[1]
	d=100
	x = x.reshape(d,)
	Y = Y.reshape(L,)
	wx = W.dot(x.T)
	alpha = np.empty([L, m])
	for i in range(L):
		for j in range(m):
			alpha[i][j] = -2 * Y[i] * kernelm(B[:, j], wx, gamma)

	#calculate beta
	beta = np.empty([L, m])
	for i in range(L):
		for j in range(m):
			summ = 0
			for l in range(m):
				summ = summ + Z[i][l] * kernelm(B[:, l], wx, gamma)
			#	# ###print("zbzbzbzb")
			#	# ###print(i, j, l)
			#	# ###print("zbzbzbzb")
			beta[i][j] = 2 * kernelm( B[:, j], W.dot(x.T), gamma) * summ
	###print(beta)
	# input("next: ")

	return alpha + beta


def testw_gradient(x, Y, W, B, Z, gamma):
	L = Z.shape[0]
	m = Z.shape[1]
	dcap = W.shape[0]
	d = W.shape[1]
	x = x.reshape(d,)
	Y = Y.reshape(L,)
	wx = W.dot(x.T)
	gamma_sq = gamma * gamma
	#calculating alpha
	alpha = np.empty([dcap, d])
	for i in range(dcap):
		for j in range(d):
			summ = np.zeros([L])
			for l in range(m):
				w_i = W[i] #.reshape(1, x.shape[1])

			#	# ###print( "wawawawaawwawa")
			#	# ###print (i)
			#	# ###print(j)
			#	# ###print(l)
			#	# ###print ("wawawawaawwawa")
			#	# ###print(kernelm(B[:, l], wx, gamma))
				## # ###print(B[:, l].shape, wx.shape, gamma)
				summ = summ + kernelm(B[:, l], wx, gamma) * (w_i.dot(x.T) - B[i,l]) * Z[: ,l]
				## # ###print(sum)
			alpha[i,j] = 4 * gamma_sq * x[0,j] * ((Y.T).dot(summ.T))
			# dada = 4 * gamma_sq * x[j] * ((Y.T).dot(sum.T))
			## # ###print(dada)

	###print(alpha)
	#calculating beta
	beta = np.empty([dcap, d])
	for i in range(dcap):
		for j in range(d):
			summ = 0
			for k in range(m):
				for l in range(m):
				#	# ###print("wbwbwbwbwb")
				#	# ###print (i)
				#	# ###print(j)
				#	# ###print(k)
				#	# ###print(l)
				#	# ###print("wbwbwbwbwb")
					summ = summ + Z[:, k].dot(Z[:, l]) * kernelm(B[:, k], wx, gamma) * x[0,j] * 2 * (-gamma_sq) * kernelm(B[:, l], wx, gamma) * (2*w_i.dot(x.T) - B[i,k] - B[i,l])
					## # ###print(sum)
			beta[i,j] = summ
	## # ###print ("end of w grad")
	#print(beta)
	
	return alpha + beta


def testb_gradient(x, Y, w, B, Z, gamma):
	m = 25
	d_cap = 15
	l = 3
	### ###print("in here :D")
	ff = np.zeros((d_cap, m))


	#x should be 100*1
	# w = 15*100

	for i in range(0,d_cap):
		for j in range(0,m):
			
			temp1 = 4* gamma * gamma * kernel(B[:,j], w.dot(x.T), gamma) * (B[i,j] - w[i].dot(x.T)) * Y.T.dot(Z[:,j]);
			### ###print(B[i,j].shape)
			ff[i,j]=temp1;

			su = 0
			
			for l in range(m):
			#	# ###print("bbbbbbbbbbbbb")
			#	# ###print(i)
			#	# ###print(j)
			#	# ###print(l)
			#	# ###print("bbbbbbbbbbbbb")				
				su = su + Z[:,l]*kernel(B[:,l], w.dot(x.T), gamma)

			## # ###print(su.shape)
			## # ###print(Z[:,j].shape)
				
			temp2 = -4 * gamma*gamma*(B[i,j] - w[i].dot(x.T))*kernel(B[:,j], w.dot(x.T), gamma)* Z[:,j].dot(su);

			ff[i,j] =ff[i,j] + temp2

	## # ###print(ff.shape)		
	return ff


def w_gradient(x,y,Y,w,B,Z,gamma,org_dim, num_prot, proj_dim):
	wx = w.dot(x.T)
	diff = -1*(B-wx)
	kernel = np.square(diff)
	kernel = np.sum(kernel, axis=0)
	kernel = -1*gamma*gamma*kernel
	kernel = np.exp(kernel)
	###print(kernel.shape)
	zker = np.multiply(kernel.T,(Z.T)).T
	# ### ###print(zker.shape)
	# ### ###print(diff.shape)
	zkerB = (zker).dot(diff.T)
	# ### ###print(zkerB.shape)
	# ### ###print(Y.shape)
	dota = (zkerB.T).dot(Y)
	# ### ###print(dota.shape)
	# ### ###print(x.shape)
	mult = 4*gamma*gamma*np.multiply(dota,x)
	###print(mult)
	# ### ###print(mult.shape)

	# quit()

	A=np.zeros((proj_dim,org_dim))
	for j in range(num_prot):
		wx = w.dot(x.T)
		Bj = np.matrix(((B.T)[j])).T
		# ###print(Bj.shape)
		# ###print(B.shape)
		Bnew = B+Bj
		# ###print(Bnew.shape)
		### ###print(B.shape)
		diffa = -1*(Bnew-2*wx)


		zcol =  np.matrix((Z.T)[j]).T
		ZZ = np.multiply(zcol, Z)
		ZZ = np.sum(ZZ, axis=0)
		# ###print(ZZ.shape)

		diffnew = Bj-wx
		# ### ###print(diffnew.shape)
		kernela = np.square(diffnew)
		# ### ###print(kernel.shape)
		kernela = np.sum(kernela, axis=0)
		# ###print(kernela.shape)
		# quit()
		kernela = -1*gamma*gamma*kernela
		kernela = np.exp(kernela)
		# ###print(kernela.shape)
		# quit()
		
		zzker = kernela*ZZ



		kernelaa = -1*gamma*gamma*kernel
		# zkerB = zker.dot(diff.T)
		zzkerker = np.multiply(kernelaa,zzker)
		##print(zzkerker.shape)
		##print(diffa.shape)
		hokage = np.multiply(zzkerker.T, diffa.T).T
		# ###print(hokage.shape)
		# ###print("prann")
		hokage = np.sum(hokage, axis=1)
		##print(hokage.shape)
		# quit()
		fin=np.dot(hokage, x)
		# quit()
		A += fin
	#print(A)
	# quit()
	return mult+A


def test(X, y, w, b, z, g):
	X =  (w.dot(X.T)).T
	# x =  (x.dot(w))
	# ### ###print(x.shape)
	# ### ###print(b.shape)
	# quit()
	#### ###print(x.shape)
	#exit()
	count=0;
	for i in range(X.shape[0]):
		x = np.matrix(X[i]).T
		# ### ###print(x.shape)
		diff = b-x
		# ### ###print(diff.shape)
		# exit()
		diff = np.square(diff)
		diff = np.sum(diff, axis = 0)
		diff = diff.T
		# ### ###print(diff.shape)
		#quit()
		diff = np.exp(-1*diff*g*g)
		# ### ###print(z.shape)
		# ### ###print("Shape of diff : " + str(diff.shape))

		# exit()

		Zdiff = np.multiply(z.T, diff)
		fsum = np.sum(Zdiff, axis = 0)
		est_y = fsum.argmax()+1
		if(y[i]==est_y):
			count=count+1
	return (count*1.0)/(X.shape[0])
	

def steplength(A, B):
	
	return 0.00001



def main():
	
	tr_data = load_svmlight_file("train.dat");#To randomly select 5000 points

	XTR = np.matrix(tr_data[0].toarray()); # Converts sparse matrices to dense
	YTR = np.matrix(tr_data[1]).T; # The trainig labels

	ts_data = load_svmlight_file("test.dat") #load_svmlight_file("train.txt");#To randomly select 5000 points ************************************CHANGE TO TEST

	XTS = np.matrix(ts_data[0].toarray()); # Converts sparse matrices to dense
	YTS = np.array(ts_data[1]).T; # The trainig labels



	labels = 3

	proj_dim = 15
	org_dim = 100
	num_prot = 200

	W = np.ones((proj_dim,org_dim))
	B = np.zeros((proj_dim,num_prot))
	Z = np.zeros((labels,num_prot))

	mean = [0]*proj_dim
	cov = np.eye(proj_dim)
	# W = (np.random.multivariate_normal(mean, cov, org_dim)).T

	# B = XTR[0:num_prot]
	# B = W.dot(B.T)

	

	proto_repr = YTR[0:num_prot]

	# ### ###print(proto_repr)

	# Z = np.array(Z)

	# for i in range(Z.shape[1]):
	# 	Z[int(proto_repr[i])-1][i] = 1


	# # ###print(Z.shape[1])
	# quit()
	#Z = 
	
	gamma = 0.2
	# np.save("Wstart", W)
	# np.save("Bstart", B)
	# np.save("Zstart", Z)
	# x = XTR[i];
	# y = YTR[i];
	# Y = np.zeros((labels,1))
	# #
	# #### ###print(int(y))
	# #### ###print(Y)
	# Y[int(y)-1][0] = 1

	# B2 = testb_gradient(x, Y, W, B, Z, gamma)
	# # ###print(B2)
	# quit()

	# W2 =  testw_gradient(x, Y, W, B, Z, gamma)
	# # ###print(W2)
	
	# Z2 =  testz_gradient(x, Y, W, B, Z, gamma)
	# # ###print(Z2)
	
	
	

	# # ###print(Z2)
	# # ###print(B2)

	# z_sum = np.zeros((labels,num_prot))
	# b_sum = np.zeros((proj_dim,num_prot))
	# w_sum = np.zeros((proj_dim,org_dim))

	# z_sum =z_sum+(10**(-1*18))
	# b_sum =b_sum+(10**(-1*18))
	# w_sum =w_sum+(10**(-1*18))

	# ###print("Z")
	# ###print(Z.shape)
	# ###print(type(Z))
	# ###print("B")
	# ###print(B.shape)
	# ###print(type(B))
	# ###print("W")
	# ###print(W.shape)
	# ###print(type(W))

	W=np.array(np.load("Wstart.npy"))
	B=np.matrix(np.load("Bstart.npy"))
	Z=np.array(np.load("Zstart.npy"))

	# # ###print("Z")
	# # ###print(Z.shape)
	# # ###print(type(Z))
	# # ###print("B.shape")
	# # ###print(B.shape)
	# # ###print("W")
	# # ###print(W.shape)
	# quit()

	print("*********************************************************************************************************")
	print(test(XTR, YTR, W, B, Z, gamma))
	print(test(XTS, YTS, W, B, Z, gamma))
	print("*********************************************************************************************************")

	# quit()
	step=0.005
	for i in range(21,XTR.shape[0]):
		x = XTR[i];
		y = YTR[i];
		Y = np.zeros((labels,1))
		#
		#### ###print(int(y))
		#### ###print(Y)
		Y[int(y)-1][0] = 1
		# for iters in range (20):
		# b_grad =testb_gradient(x, Y, W, B, Z, gamma)
		# w_grad2 =w_gradient(x,y,Y,W,B,Z,gamma,org_dim, num_prot, proj_dim)
		
		# w_grad1 =testw_gradient(x,Y,W,B,Z,gamma)
		# ###print(w_grad1)
		# ###print(w_grad2)

		# quit()

		# z_grad =z_gradient(x, y, Y, W, B, Z, gamma)
		# ###print("zgrad")

		# ###print(z_grad)
		# quit()
		# z_sum = np.square(z_grad)
		# z_sum =z_sum+np.square(z_grad)
		# z_sum =z_sum+(10**(-1*18))
		# # ###print(steplength(z_grad, z_sum))
		# quit()
		# b_grad =b_gradient(x, y, Y, W, B, Z, gamma, org_dim, num_prot, proj_dim)
		# ###print("bgrad")
		# ###print(b_grad)
		# w_grad =w_gradient(x,y,Y,W,B,Z,gamma,org_dim, num_prot, proj_dim)
		# ###print("wgrad")
		# ###print(w_grad)
		## # ###print(z_gradient(x, y, Y, W, B, Z, gamma))
		## # ###print(b_gradient(x, y, Y, W, B, Z, gamma, org_dim, num_prot, proj_dim))
		## # ###print(w_gradient(x,y,Y,W,B,Z,gamma,org_dim, num_prot, proj_dim))
		if(i==500):
			print("*********************************************************************************************************")
			print(test(XTR[0:i], YTR[0:i], W, B, Z, gamma))
			print(test(XTS, YTS, W, B, Z, gamma))
			print("*********************************************************************************************************")


		for epochs in range(1):
			z_grad =z_gradient(x, y, Y, W, B, Z, gamma)
			b_grad =b_gradient(x, y, Y, W, B, Z, gamma, org_dim, num_prot, proj_dim)
			w_grad =w_gradient(x,y,Y,W,B,Z,gamma,org_dim, num_prot, proj_dim)
			# z_step = steplength()
			# b_step = steplength(b_grad, b_sum)
			# w_step = steplength(w_grad, w_sum)
			Z = Z - step*z_grad
			B = B - step*b_grad
			W = W - step*w_grad
		# quit()

		# z_sum =z_sum+np.square(z_grad)
		# b_sum =b_sum+np.square(b_grad)
		# w_sum =w_sum+np.square(w_grad)
		## # ###print(z_sum.shape);

		### ###print(z_sum)
		#quit()

		# if(i==0):
		# 	z_sum =z_sum+(10**(-1*18))
		# 	b_sum =b_sum+(10**(-1*18))
		# 	w_sum =w_sum+(10**(-1*18))


		# z_step = steplength(z_grad, z_sum)
		# b_step = steplength(b_grad, b_sum)
		# w_step = steplength(w_grad, w_sum)

		# Z = Z - z_step*z_grad
		# B = B - b_step*b_grad
		# W = W - w_step*w_grad

		# ###print("Z")
		# ###print(Z)
		# ###print("B")
		# ###print(B)
		# ###print("W")
		# ###print(W)
		# np.save("Z", Z)
		# np.save("B", B)
		# np.save("W", W)
		# quit()

		## # ###print(z_gradient(x, y, Y, W, B, Z, gamma))
		## # ###print(b_gradient(x, y, Y, W, B, Z, gamma, org_dim, num_prot, proj_dim))
		## # ###print(w_gradient(x,y,Y,W,B,Z,gamma,org_dim, num_prot, proj_dim))
		
		## # ###print(Z)
		## # ###print(B)
		## # ###print(W)
		# if(i%5 == 0):
			## # ###print(z_gradient(x, y, Y, W, B, Z, gamma))
			## # ###print(b_gradient(x, y, Y, W, B, Z, gamma, org_dim, num_prot, proj_dim))
			## # ###print(w_gradient(x,y,Y,W,B,Z,gamma,org_dim, num_prot, proj_dim))
			# qwerty=input("Press to continue");

		#	# ###print(i)
		#	# ###print(Z)
		#	# ###print(B)
		#	# ###print(W)
		if(i%5000==0):
			print("*********************************************************************************************************")
			print(test(XTR[0:i], YTR[0:i], W, B, Z, gamma))
			print(test(XTS, YTS, W, B, Z, gamma))
			print("*********************************************************************************************************")
			#print(W)
			# qwerty=raw_input("Press to continue");

	print("*********************************************************************************************************")
	print(test(XTR[0:i], YTR[0:i], W, B, Z, gamma))
	print(test(XTS, YTS, W, B, Z, gamma))
	print("*********************************************************************************************************")

	np.save("Wend", W)
	np.save("Bend", B)
	np.save("Zend", Z)


if __name__ == '__main__':
	main()
