import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
# import plotly.graph_objects as go

# Create Hessian Matrix
def createHessian(point):
	return np.array([
		[(1-point[0]-point[1])**-2 + point[0]**-2, (1-point[0]-point[1])**-2],
		[(1-point[0]-point[1])**-2, (1-point[0]-point[1])**-2 + point[1]**-2]])

# Create Gradient Matrix
def createGradient(point):
	return np.array([(1-point[0]-point[1])**-1 - point[0]**-1, (1-point[0]-point[1])**-1 - point[1]**-1])

# Get Random Point under constraints
def getPoint():
	x_ = np.random.uniform(0,1)
	y_ = np.random.uniform(0,1-x_)
	if x_> 0 and y_ > 0 and (x_ + y_ < 1):
		return [x_, y_]

# Get Value of the function
def functionValue(xdash, ydash):
	try:
		# print(-math.log(1-xdash-ydash)-math.log(xdash)-math.log(ydash))
		return -math.log(1-xdash-ydash)-math.log(xdash)-math.log(ydash)
	except:
		print("Error", xdash, ydash)

def gradDescent(w_, n, eta, epsilon_):
	print("w0: {}".format(w_))

	c=0 #counter
	endPoints = []
	lastPoint=0
	curVal = []
	tempW = 0
	while c<n:
		x_ = w_[0]
		y_ = w_[1]
		g = np.array([(1-x_-y_)**-1 - x_**-1, (1-x_-y_)**-1 - y_**-1])
		# Calculating Gradient Descent
		tempW = w_ - eta * g
		if isinstance(lastPoint, np.ndarray):
			if abs(np.linalg.norm(tempW-lastPoint)) < epsilon_:
				break

		if tempW[0]>0 and tempW[1]>0 and sum(tempW)<1:
			w_=tempW
			print("w__",w_)
			endPoints.append(w_)
			curVal.append(functionValue(w_[0],w_[1]))
			lastPoint = w_[:]
			# print("w{}: {}  ______  {}".format(c+1, tempW[-1], functionValue(tempW[0],tempW[1])))
		else:
			# print("****w{}: {}  ______  {}".format(c+1, tempW[-1], functionValue(tempW[0],tempW[1])))
			w_ = np.array(getPoint())
			lastPoint=0
			c+=1

		c+=1
	return endPoints, curVal, c

def NewtonMethod(w_, n, eta, epsilon_):
	print("w0: {}".format(w_))

	c=0 #counter
	endPoints = []
	lastPoint=0
	curVal = []
	tempW = 0
	H = createHessian(w_)
	print("H", H)
	while c<n:
		x_ = w_[0]
		y_ = w_[1]
		g = np.array([(1-x_-y_)**-1 - x_**-1, (1-x_-y_)**-1 - y_**-1])
		# Calculating Newton Method
		tempW = w_ - eta * np.matmul(np.linalg.pinv(H), g)
		# print(w_, tempW)
		if isinstance(lastPoint, np.ndarray):
			if abs(np.linalg.norm(tempW-lastPoint)) < epsilon_:
				break
		if tempW[0]>0 and tempW[1]>0 and sum(tempW)<1:
			w_=tempW
			endPoints.append(w_)
			curVal.append(functionValue(w_[0],w_[1]))
			lastPoint = w_[:]
			# print("w{}: {}  ______  {}".format(c+1, tempW[-1], functionValue(tempW[0],tempW[1])))
		else:
			# print("**** w{}: {}  ______  {}".format(c+1, tempW[-1], functionValue(tempW[0],tempW[1])))
			w_ = np.array(getPoint())
			lastPoint=0
			c+=1

		c+=1
	return endPoints, curVal, c


if __name__ == '__main__':
	# f(x,y) = -log(1-x-y)-log(x)-log(y)
	# contraint x+y < 1 ; x > 0 ; y > 0

	# 1 (a)
	# df/dx = (1-x-y)^-1 - x^-1
	# df/dy = (1-x-y)^-1 - y^-1
	# d^2f/dx^2 = (1-x-y)^-2 + x^-2
	# d^2f/dx^2 = (1-x-y)^-2 + y^-2
	# d^2f/dxy = d^2f/dyx = (1-x-y)^-2

	# gradient g = [df/dx df/dy]^T [^T implies Transpose]

	# lRate = 1

	w0 = getPoint()
	# print("Point: {}".format(w0), sum(w0))

	# number_of_epochs = 50

	# w = gradDescent(np.array(w0[:]), number_of_epochs, lRate)

	# print(w)
	epsilon = 10**-5
	#changing Learning Rate
	lRate = 0.01

	number_of_epochs = 500

	w, valList, n = gradDescent(np.array(w0[:]), number_of_epochs, lRate, epsilon)
	print("n:", n)
	print(w[np.argmin(np.array(valList))], min(valList))

	xtemp, ytemp = zip(*w)
	# print(xtemp)
	# print(ytemp)
	# print(w)

	# Plot using Matplotlib 3d
	ax = plt.axes(projection='3d')
	ax.scatter3D(xtemp, ytemp, valList, c=valList, cmap='terrain');
	plt.show()
	
	# Plot using Plotly 3d
	# fig = go.Figure([go.Scatter3d(x=xtemp, y=ytemp, z=valList,text=[str(i) for i in range(len(valList))], mode='markers')])
	# fig.update_layout(scene = dict(
 #        xaxis_title='X',
 #        yaxis_title='Y',
 #        zaxis_title='Function Value',
 #        aspectratio=dict(x=1, y=1, z=1)),
	# 	title="Trajectory on Minimization using Gradient Descent ",
	#     # xaxis_title="X",
	#     # yaxis_title="Y",
	#     # zaxis_title="Function Value",
	#     # legend_title="Legend Title",
	# 	font=dict(
	# 		family="Courier New",
	# 		size=13,
	# 		color="Mediumorchid"
	# 	)
	# )
	# fig.update_yaxes(range = [0,1])
	# fig.update_xaxes(range = [0,1])


	# fig.show()

	plt.title("Energies by Iteration Gradient Descent")
	iteration_no=[0]
	iteration_no.extend([j+1 for j in range(len(valList))])
	plt.plot(iteration_no, [functionValue(w0[0], w0[1])]+valList)
	plt.xlabel("Epochs")
	plt.ylabel("f(X,Y)")

	# plt.xticks(range(len(iteration_no)), list(mis_epoch.keys()))
	plt.show()


	# epsilon = 10**-4
	lRate = 1
	number_of_epochs = 12000
	w, valList, n = NewtonMethod(np.array(w0[:]), number_of_epochs, lRate, epsilon)
	print("n:", n)
	print(w[np.argmin(np.array(valList))], min(valList))

	xtemp, ytemp = zip(*w)
	# print(xtemp)
	# print(ytemp)
	# print(w)

	# Plot using Matplotlib 3d
	ax = plt.axes(projection='3d')
	ax.scatter3D(xtemp, ytemp, valList, c=valList, cmap='terrain');
	plt.show()
	
	# Plot using Plotly 3d
	# fig = go.Figure([go.Scatter3d(x=xtemp, y=ytemp, z=valList, text=[str(i) for i in range(len(valList))], mode='markers')])
	# fig.update_layout(scene = dict(
 #        xaxis_title='X',
 #        yaxis_title='Y',
 #        zaxis_title='Function Value'),
	# 	title="Trajectory on Minimization using Newton Method  ",
	#     # xaxis_title="X",
	#     # yaxis_title="Y",
	#     # zaxis_title="Function Value",
	#     # legend_title="Legend Title",
	# 	font=dict(
	# 		family="Courier New",
	# 		size=13,
	# 		color="Blue"
 #        )
 #    )
	# fig.update_yaxes(range = [0,1])
	# fig.update_xaxes(range = [0,1])
	# fig.show()

	plt.title("Energies by Iteration Newton Method")
	iteration_no=[0]
	iteration_no.extend([j+1 for j in range(len(valList))])
	plt.plot(iteration_no, [functionValue(w0[0], w0[1])]+valList)
	plt.xlabel("Epochs")
	plt.ylabel("f(X,Y)")
	plt.show()

	print(functionValue(0.3333, 0.3333))