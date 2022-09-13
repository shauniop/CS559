import math
import numpy as np
from matplotlib import pyplot as plt
from time import time

# Activation Functions
def actFxn(inp):
	return np.tanh(inp)

def actFxnDash(inp):
	return 1 - np.tanh(inp)**2

def actFxn2(finalInp):
	return finalInp

def actFxnDash(finalInp):
	return 1



if __name__ == '__main__':
	#Setting random seed
	# np.random.seed(1597)

	# Draw n = 300 real numbers uniformly at random on [0, 1], call them x1, . . . , xn
	n = 300
	x = np.random.uniform(0, 1, n)

	# Draw n real numbers uniformly at random on [−0.1, 0.1], call them ν1, . . . , νn
	v = np.random.uniform(-0.1, 0.1, n)

	# Let di = sin(20xi) + 3xi + νi, i = 1, . . . , n
	d = np.sin(20*x) + 3*x + v
	# print(d)

	# Plot the points (xi, di), i = 1, . . . , n
	plt.scatter(x, d)
	# plt.show()

	N = 24

	w1 = np.random.uniform(-10,10, N) #24 WEIGHTS FIRST LAYER
	w2 = np.random.uniform(-10,10, N) #24 WEIGHTS SECOND LAYER

	b1 = np.random.uniform(-1, 1, N) # 24 WEIGHT BIAS
	b2 = np.random.uniform(-1, 1, 1)[0] # FINAL SINGLE BIAS 

	# eta = 10
	# eta = 1
	# eta = 0.1
	eta = 0.01
	# eta = 0.001
	break_mse = 10*-1
	start = time()
	mse_list = [np.mean(d-[np.dot(w2.T,np.tanh(w1*xj+b1))+b2 for xj in x])**2]
	# print(mse_list)

	for i in range(10):
		for j in range(n):
			xi=x[j]
			di=d[j]
			# Implement feedforward network
			a1 = np.tanh(w1*xi+b1)
			a2 = np.dot(w2.T,a1)+b2

			y=a2
		
			# Implement feedback network
			error = di - y
			b2 = b2 + eta * error
			w2 = w2 + eta * a1 * error
			b1 = b1 + eta * error * w2 * actFxnDash(w1*xi+b1)
			w1 = w1 + eta * xi * error * w2 * actFxnDash(w1*xi+b1)
			ydash = [np.dot(w2.T,np.tanh(w1*x[k]+b1))+b2 for k in range(n)]
			mse_list.append(np.mean(d-y)**2)

			if mse_list[-1]<break_mse:
				y=[np.dot(w2.T,np.tanh(w1*xi+b1))+b2 for xi in x]
				c = zip(x,y)
				c=sorted(c, key = lambda x:x[0])
				x, y =zip(*c)
				plt.plot(x, y, c='r')
				plt.show()

				plt.plot(range(len(mse_list[1:])),mse_list[1:])

				plt.show()
				sys.exit()

			if mse_list[-2]<mse_list[-1]:
				eta*=0.9


	y=[np.dot(w2.T,np.tanh(w1*xi+b1))+b2 for xi in x]
	print(x.shape)
	c = zip(x,y)
	c=sorted(c, key = lambda x:x[0])
	x, y =zip(*c)
	plt.plot(x, y, c='r')
	plt.show()

	plt.plot(range(len(mse_list[1:])),mse_list[1:])

	plt.show()