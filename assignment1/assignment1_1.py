import numpy as np 
import csv
# Linear Regression Model
def predict(X, W):
	y_predict = X.dot(W)
	return y_predict

def L2Cost(X, W, y_target):
	cost = 0.0
	M,N = X.shape()
	grads = np.array(W.shape())
	scores = X.dot(W)
	
	cost = np.sum(np.square(scores - y_target))/N

	grads = X.T.dot(scores - y_target)

	return cost, grads

def Stochastic_Gradient_Descent(X, y_target, W, epochs=100, learning_rate=10e-5):
	for i in range(epochs):
		# To prevent overfitting of data
		for j in range(10):


# Load Data and create Training and Test data
