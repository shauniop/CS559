# -*- coding: utf-8 -*-
# Python 3.9.1
# numpy==1.21.2
# matplotlib==3.4.3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

#Setting seed for np.random
np.random.seed(123)

# Calculating miscalculations
def miscalc(S, wts, wts_init):
	mis=0
	for indx in S:
		if pred(indx[0], indx[1], wts[0], wts[1], wts[2]) and not pred(indx[0], indx[1], wts_init[0], wts_init[1], wts_init[2]):
			mis+=1
		elif not pred(indx[0], indx[1], wts[0], wts[1], wts[2]) and pred(indx[0], indx[1], wts_init[0], wts_init[1], wts_init[2]):
			mis+=1
	return mis

# Calculating output / input to unit function
def calc(_x1, _x2, _w0, _w1, _w2):
	return	_w0 + _x1*_w1 + _x2*_w2

# To check the output satisfies function >= 0
def pred(_x1, _x2, _w0, _w1, _w2):
	return calc(_x1, _x2, _w0, _w1, _w2) >= 0

# Step Function implementation
def step(x):
	return 1 if x>=0 else 0

# Perceptron Training Algorithm
def PTA(weights, weights_, S, lRate):
	epochNo = 0
	mis=0
	mis_epoch={}
	# run till misclassification is 0
	while True:
		epochNo+=1
		mis=0
		for pt in S:
			original = step(calc(pt[0], pt[1], weights[0], weights[1], weights[2]))
			new = step(calc(pt[0], pt[1], weights_[0], weights_[1], weights_[2]))
			diff = original - new
			if diff:
				mis+=1
			# updating weight
			weights_[0] +=  lRate * diff * 1
			for indx, w in enumerate(weights_[1:]):
				weights_[indx+1] += lRate * diff * pt[indx]
		mis_epoch[epochNo] = mis
		if epochNo==1:
			print("Number of misclassifications for weights[ w0'', w1'', w2''] :{}".format(mis))
		if mis==0:
			# print(mis_epoch)
			#1 i.
			plt.title("Epoch number vs the number of misclassifications [η={}]".format(lRate))
			plt.bar(range(len(mis_epoch)), list(mis_epoch.values()), align='center')
			plt.xticks(range(len(mis_epoch)), list(mis_epoch.keys()))
			plt.show()
			return weights_, epochNo

def PTA_analysis(n):
	#Pick (your code should pick it) w0 uniformly at random on [−1/4,1/4].
	#1 a.
	w0 = np.random.uniform(-0.25,0.25)

	# Pick w1 uniformly at random on [−1, 1].
	#1 b.
	w1 = np.random.uniform(-1,1)

	# Pick w2 uniformly at random on [−1, 1].
	#1 c.
	w2 = np.random.uniform(-1,1)

	print("Optimal weights")
	print("w0 = {}, w1 = {}, w2 = {}".format(w0,w1,w2))

	# Pick n vectors x1, . . . , xn independently and uniformly at random on [−1, 1]^2 , call the collection of these vectors S.
	#1 d.
	# n = 100/1000 [inputing from function]

	S = np.random.uniform(low = -1, high = 1, size=(n,2))

	#1 e.
	#1 f.
	# S0, S1 ∈ S
	S0 = []
	S1 = []
	pts1_x = []
	pts1_y = []
	pts2_x = []
	pts2_y = []
	for indx in S:
		if pred(indx[0], indx[1], w0, w1, w2):
			S1.append([indx[0]] + [indx[1]])
			pts1_x.append(indx[0])
			pts1_y.append(indx[1])
		else:
			S0.append([indx[0]] + [indx[1]])
			pts2_x.append(indx[0])
			pts2_y.append(indx[1])

	#1 g.
	plt.scatter(pts1_x, pts1_y, c='b', marker='o', label='S1')
	plt.scatter(pts2_x, pts2_y, c='r', marker='x', label='S2')
	x = np.linspace(-1,1,100)
	y = (-w0 - w1 * x) / w2

	patches.Patch()
	plt.plot(x, y, '-r', label='Separator')
	plt.title('Graph of w0 + w1x1 + w2x2 = 0 [N = {}]'.format(n))

	plt.xlabel('x1', color='#1C2833')
	plt.ylabel('x2', color='#1C2833')
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()

	#1 h.
	# Use the training parameter η = 1 and 
	eta = 1

	#  Pick w0', w1', w2' independently and uniformly at random on [−1, 1]
	w0_ = np.random.uniform(-1,1)
	w1_ = np.random.uniform(-1,1)
	w2_ = np.random.uniform(-1,1)

	print("w0' = {}, w1' = {}, w2' = {}".format(w0_,w1_,w2_))

	mis = miscalc(S, [w0_, w1_, w2_], [w0, w1, w2])

	print("Miscalculations {}".format(mis))

	w0__,w1__,w2__ = [],[],[]

	weights = [w0, w1, w2]

	weights_ = [w0_, w1_, w2_]
	weights2_ = weights_[:]

	# Learning rate as 1
	lRate = 1
	weights_ = weights2_[:]
	weights__, epochs= PTA(weights, weights_, S, lRate)
	print("Epochs for n = {} & η = {}: {}".format(n,lRate,epochs))
	print("Final weights:", weights__)

	#1 j.
	# Learning rate as 10
	lRate = 10
	weights_ = weights2_[:]
	weights__, epochs= PTA(weights, weights_, S, lRate)
	print("Epochs for n = {} & η = {}: {}".format(n,lRate,epochs))
	print("Final weights:", weights__)

	#1 k.
	# Learning rate as 0.1
	lRate = 0.1
	weights__, epochs= PTA(weights, weights2_, S, lRate)
	print("Epochs for n = {} & η = {}: {}".format(n,lRate,epochs))
	print("Final weights:", weights__)

if __name__ == '__main__':
	PTA_analysis(n = 100) # PTA Analysis for vector size 100
	#1 n.
	PTA_analysis(n = 1000) # PTA Analysis for vector size 1000