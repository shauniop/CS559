import numpy as np
from matplotlib import pyplot as plt

def getPoint():
	x_ = np.random.uniform(0,1)
	y_ = np.random.uniform(0,1-x_)
	if x_> 0 and y_ > 0 and (x_ + y_ < 1):
		return [x_, y_]

def createGradient(point, xi, yi):
	return (np.array([
        sum((-2 *[yi[i] - point[0] - point[1] * xi[i] for i in range(50)])),
        sum((-2 *[(yi[i] - point[0] - point[1] * xi[i]) * xi[i] for i in range(50)]))
    ])).reshape(2, 1)

def functionVal(x_, w0_, w1_):
	return w0_ + w1_*x



def gradDescent(w_, n, eta, epsilon_, x__, y__):
	print("w0: {}".format(w_))

	c=0 #counter
	endPoints = []
	lastPoint=0
	# curVal = []
	tempW = 0
	while c<n:
		x_ = w_[0]
		y_ = w_[1]
		g = createGradient(w_, x__, y__)
		# Calculating Gradient Descent
		tempW = w_ - eta * g
		if isinstance(lastPoint, np.ndarray):
			if abs(np.linalg.norm(tempW-lastPoint)) < epsilon_:
				break
		w_=tempW
		lastPoint=w_[:]
		endPoints.append(w_)

		c+=1
		if c%50==0:
			print(c)
	return endPoints, c



if __name__ == '__main__':
	x_i = [i + 1 for i in range(50)]
	y_i=[]
	for i in x_i:
		y_i.append(i+np.random.uniform(-1, 1))

	# Linear Square Fit initialization
	X=[]
	X.append(np.array([1] * 50))
	X.append(np.array(x_i))
	X = np.array(X).reshape(2, 50)
	Y=np.array(y_i).reshape(1, 50)

	w = np.matmul(Y, np.linalg.pinv(X))
	print("w0: {}\nw1: {}\n".format(w[0][0], w[0][1]))



	# Compute weights using gradient descent
	w0 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
	epsilon = 10**-15
	lRate = 0.01

	number_of_epochs = 5000



	w, n = gradDescent(np.array(w0[:]), number_of_epochs, lRate, epsilon, x_i, y_i)
	print(w[-1])
	print(n)
	x = range(50)
	for i in x:
		plt.scatter(x_i,y_i)
		plt.plot(x, functionVal(x, w[-1][0],w[-1][1]),'-r', label='Separator')
	plt.title("Fitting using Gradient")

	plt.show()