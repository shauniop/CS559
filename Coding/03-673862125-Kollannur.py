# -*- coding: utf-8 -*-
# Python 3.9.1
# numpy==1.21.2
# matplotlib==3.4.3

import gzip
import struct
import numpy as np
from matplotlib import pyplot as plt
from time import time as tm


def print2D_0_1(z):
	for _ in z:
		for __ in _:
			print(0 if __ < 0 else 1, end="")
		print()

def print2D(z):
	for _ in z:
		for __ in _:
			print(__, end="")
		print()


# Step Function implementation
def step(x):
	# print("prestep", x)
	return np.array([1 if i==np.argmax(x.reshape(10)) else 0 for i in range(10)])

# Calculation
def calc(weights__, x__):
	return np.matmul(weights__, x__)

# Perceptron Training Algorithm
def PTA(label, weights_, imageArray_, lRate, epsilon_, N_):
	epochNo = 0
	mis=0
	mis_epoch={}
	# run till misclassification is 0
	start=tm()
	while True:
		epochNo+=1
		mis=0
		for imgInput, original in zip(imageArray_, label):
			imgInput_= imgInput.flatten().reshape(784, 1)
			# print2D(weights_)
			# print2D(imgInput_)
			# print(weights_.shape, imgInput_.shape)
			new = np.argmax(calc(weights_, imgInput_).reshape(10))
			diff = original - new
			if diff:
				mis+=1
		mis_epoch[epochNo] = mis
		# if tm()-start > 600:
		# 	return weights_, epochNo, mis_epoch
		if epochNo%10==0:
			print("{} > EpochNo {} Errors: {}".format(tm()-start, epochNo,mis))
		if mis/N_ <= epsilon_:
			return weights_, epochNo, mis_epoch		
		# updating weight
		for imgInput, original in zip(imageArray_, labels):
			imgInput_= imgInput.flatten().reshape(784, 1)
			localField = calc(weights_, imgInput_)
			new = step(localField)
			original_ = np.array([1 if i==original else 0 for i in range(10)])
			diff = (original_ - new).reshape(10,1)
			weights_ += lRate * np.matmul( diff,np.transpose(imgInput_))
		
		
# Validate Weights
def validateWeightsOnTestingSet(imageArray_, labels, weights_):
	mis=0
	for imgInput, original in zip(imageArray_, labels):
		imgInput_= imgInput.flatten().reshape(784, 1)
		new = np.argmax(calc(weights_, imgInput_).reshape(10))
		diff = original - new
		if diff:
			mis+=1
	return mis


############ Extracting Data from MNIST ################

with gzip.open("MNIST/train-images-idx3-ubyte.gz","r") as trainImageFile:
	# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
	# [offset] [type]          [value]          [description] 
	# 0000     32 bit integer  0x00000803(2051) magic number
	# 0004     32 bit integer  60000            number of images 
	# 0008     32 bit integer  28               number of rows 
	# 0012     32 bit integer  28               number of columns
	# 0016     unsigned byte   ??               pixel
	# 0017     unsigned byte   ??               pixel
	# ........
	# xxxx     unsigned byte   ??               pixel


	# Train Image Data Info Extraction
	print("** Train Image Data Info Extraction")
	# 32 bit / 4 bytes magic number
	magic_number = int.from_bytes(trainImageFile.read(4), "big")
	print("magic_number:",  magic_number) # 2051
	# 32 bit / 4 bytes number of images
	number_images = int.from_bytes(trainImageFile.read(4), "big")	
	print("number_images:", number_images) # 60000
	# 32 bit / 4 bytes number of rows
	row_count = int.from_bytes(trainImageFile.read(4), "big")	
	print("row_count:", row_count) # 28
	# 32 bit / 4 bytes number of columns
	col_count = int.from_bytes(trainImageFile.read(4), "big")	
	print("col_count:", col_count) # 28

	# Store Image
	imageBuf = trainImageFile.read(row_count * col_count * number_images)
	imageArray_ = np.frombuffer(imageBuf, dtype=np.uint8).astype(np.float32)
	imageArray = imageArray_.reshape(number_images, row_count, col_count)



with gzip.open("MNIST/train-labels-idx1-ubyte.gz","r") as trainLabel:
	# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
	# [offset] [type]          [value]          [description]
	# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
	# 0004     32 bit integer  60000            number of items
	# 0008     unsigned byte   ??               label
	# 0009     unsigned byte   ??               label
	# ........
	# xxxx     unsigned byte   ??               label

	# Train Label Data Info Extraction
	print("\n** Train Label Data Info Extraction")
	# 32 bit / 4 bytes magic number
	magic_number = int.from_bytes(trainLabel.read(4), "big")
	print("magic_number:",  magic_number) # 2049
	# 32 bit / 4 bytes number of images
	number_items = int.from_bytes(trainLabel.read(4), "big")	
	print("number_items:", number_items)

	# Store labels
	labels = np.frombuffer(trainLabel.read(),np.uint8)

with gzip.open("MNIST/t10k-images-idx3-ubyte.gz","r") as testImageFile:
	# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
	# [offset] [type]          [value]          [description]
	# 0000     32 bit integer  0x00000803(2051) magic number
	# 0004     32 bit integer  10000            number of images
	# 0008     32 bit integer  28               number of rows
	# 0012     32 bit integer  28               number of columns
	# 0016     unsigned byte   ??               pixel
	# 0017     unsigned byte   ??               pixel
	# ........
	# xxxx     unsigned byte   ??               pixel

	# Test Image Data Info Extraction
	print("\n** Test Image Data Info Extraction")
	# 32 bit / 4 bytes magic number
	magic_number_test = int.from_bytes(testImageFile.read(4), "big")
	print("magic_number:",  magic_number_test) # 2051
	# 32 bit / 4 bytes number of images
	number_images_test = int.from_bytes(testImageFile.read(4), "big")	
	print("number_images:", number_images_test) # 10000
	# 32 bit / 4 bytes number of rows
	row_count_test = int.from_bytes(testImageFile.read(4), "big")	
	print("row_count:", row_count_test) # 28
	# 32 bit / 4 bytes number of columns
	col_count_test = int.from_bytes(testImageFile.read(4), "big")	
	print("col_count:", col_count_test) # 28

	# Store Image
	testimageBuf = testImageFile.read(row_count_test * col_count_test * number_images_test)
	testImageArray_ = np.frombuffer(testimageBuf, dtype=np.uint8).astype(np.float32).reshape(number_images_test, row_count_test*col_count_test)
	testImageArray = testImageArray_.reshape(number_images_test, row_count_test, col_count_test)

with gzip.open("MNIST/t10k-labels-idx1-ubyte.gz","r") as testLabel:
	# TESTING SET LABEL FILE (test-labels-idx1-ubyte):
	# [offset] [type]          [value]          [description]
	# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
	# 0004     32 bit integer  60000            number of items
	# 0008     unsigned byte   ??               label
	# 0009     unsigned byte   ??               label
	# ........
	# xxxx     unsigned byte   ??               label

	# Test Label Data Info Extraction
	print("\n** Test Label Data Info Extraction")
	# 32 bit / 4 bytes magic number
	magic_number_test_labels = int.from_bytes(testLabel.read(4), "big")
	print("magic_number_test_labels:",  magic_number_test_labels) # 2049
	# 32 bit / 4 bytes number of images
	number_items_test_labels = int.from_bytes(testLabel.read(4), "big")	
	print("number_items_test_labels:", number_items_test_labels)

	# Store labels
	testlabels = np.frombuffer(testLabel.read(),np.uint8)
	print()

############ Initializing Random Weights ################
W = []	
for i in range(10):
	W.append([])
	for j in range(28*28):
		W[i].append(np.random.uniform(-1,1))
W = np.array(W)

############ n = 50, η = 1 ################
# Initialisation for 1 (f)
N = 50
learningRate = 1
epsilon = 0

#1 (d)
wt_, epNo, mis_epoch_ = PTA(labels[:N], W[:], imageArray[:N], learningRate, epsilon, N)
np.save("weight_50.npy", wt_) # saving weight
print(wt_, epNo, mis_epoch_)

# 1 (f) plot
plt.title("Epoch number vs the number of misclassifications [η={}]".format(learningRate))
plt.bar(range(len(mis_epoch_)), list(mis_epoch_.values()), align='center')
plt.xticks(range(len(mis_epoch_)), list(mis_epoch_.keys()))
plt.show()

# 1 (e)
N_test = 10000 # entire test db
# print(testImageArray_.shape)
error_test = validateWeightsOnTestingSet(testImageArray_[:N_test], testlabels[:N_test], wt_)
print("Error: {}\nAccuracy: {:.2f}%".format(error_test, (1-error_test/N_test)*100))


############ n = 1000, η = 1 ################
# Initialisation for 1 (g)   
N = 1000
learningRate = 1
epsilon = 0xxf

#1 (d)
wt_, epNo, mis_epoch_ = PTA(labels[:N], W[:], imageArray[:N], learningRate, epsilon, N)
np.save("weight_1000.npy", wt_) # saving weight
print(wt_, epNo, mis_epoch_)

# 1 (f) plot
plt.title("Epoch number vs the number of misclassifications [η={}]".format(learningRate))
plt.bar(range(len(mis_epoch_)), list(mis_epoch_.values()), align='center')
plt.xticks(range(len(mis_epoch_)), list(mis_epoch_.keys()))
plt.show()

# 1 (e)
N_test = 10000 # entire test db
# print(testImageArray_.shape)
error_test = validateWeightsOnTestingSet(testImageArray_[:N_test], testlabels[:N_test], wt_)
print("Error: {}\nAccuracy: {:.2f}%".format(error_test, (1-error_test/N_test)*100))


# ############ n = 60000, η = 1 ################
# # Initialisation for 1 (h)   
N = 60000
learningRate = 1
epsilon = 0

#1 (d)
wt_, epNo, mis_epoch_ = PTA(labels[:N], W[:], imageArray[:N], learningRate, epsilon, N)
print(wt_, epNo, mis_epoch_)
# 1 (f) plot
plt.title("Epoch number vs the number of misclassifications [η={}]".format(learningRate))
plt.bar(range(len(mis_epoch_)), list(mis_epoch_.values()), align='center')
plt.xticks(range(len(mis_epoch_)), list(mis_epoch_.keys()))
plt.show()

# 1 (e)
N_test = 10000 # entire test db
# print(testImageArray_.shape)
error_test = validateWeightsOnTestingSet(testImageArray_[:N_test], testlabels[:N_test], wt_)
print("Error: {}\nAccuracy: {:.2f}%".format(error_test, (1-error_test/N_test)*100))


############ n = 60000, η = 1 ################
# Initialisation for 1 (i)  
W=np.load("weight_1000.npy") 
# W=np.load("weight_50.npy") 
N = 60000
learningRate = 1
epsilon = 0.1

#1 (d)
wt_, epNo, mis_epoch_ = PTA(labels[:N], W[:], imageArray[:N], learningRate, epsilon, N)
print(wt_, epNo, mis_epoch_)
# 1 (f) plot
plt.title("Epoch number vs the number of misclassifications [η={}]".format(learningRate))
plt.bar(range(len(mis_epoch_)), list(mis_epoch_.values()), align='center')
plt.xticks(range(len(mis_epoch_)), list(mis_epoch_.keys()))
plt.show()

# 1 (e)
N_test = 10000 # entire test db
# print(testImageArray_.shape)
error_test = validateWeightsOnTestingSet(testImageArray_[:N_test], testlabels[:N_test], wt_)
print("Error: {}\nAccuracy: {:.2f}%".format(error_test, (1-error_test/N_test)*100))