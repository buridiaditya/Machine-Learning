import numpy as np

class NeuralNetwork(object):
	def __init__(self,input_dim,hidden_dims,num_classes,activation_function,weight_scale=10e-2):
		L = hidden_dims.shape
		self.params = {}
		self.activation = activation_function
		self.hidden_layers = L
		for i in range(L+1):
			indexW = 'W' + str(i)
			indexb = 'b' + str(i)
			if i == 0
				self.params[indexW] = np.random.normal(0,weight_scale,(input_dim,hidden_dims[i]))
				self.params[indexb] = np.random.normal(0,weight_scale,(hidden_dims[i]))
			else if i == L:
				self.params[indexW] = np.random.normal(0,weight_scale,(hidden_dims[i-1],num_classes))
				self.params[indexb] = np.random.normal(0,weight_scale,(num_classes))
			else : 
				self.params[indexW] = np.random.normal(0,weight_scale,(hidden_dims[i-1],hidden_dims[i]))
				self.params[indexb] = np.random.normal(0,weight_scale,(hidden_dims[i]))

	def linear(self,X,W,b):
		value = X.dot(W) + b
		return value

	def act_tanh(self,X):
		value = np.tanh(X)
		return value

	def act_sigmoid(self,X):
		value = 1/(1+np.exp(-X))
		return value

	def dtanh(self,dX,out):
		return dX*(1 - np.square(out))

	def dsigmoid(self,dX,out):
		return dX*(out)*(1-out)

	def dlinear(self,dX,W,X):
		dW = X.T.dot(dX)
		dx = dX.dot(W.T)
		db = np.ones(X.shape[0]).dot(dX)
		return dW, db, dx

	def loss_1(self,X,y_target):
		N,_  = X.shape
		X_i = X
		backup = []
		for i in range(self.hidden_layers + 1):
			indexW = 'W' + str(i)
			indexb = 'b' + str(i)	
			W = self.params[indexW] 
			b = self.params[indexb] 
			X_old = X_i
			X_i = linear(X_i,W,b)
			# Use any activation fucntion
			X_i = act_tanh(X_i)
			backup.append((X_old,X_i))
			# X_i = act_sigmoid(X_i)

		# For sigmoid activation function
		# y_target[y_target == 0] = -1
		scores = X_i
		cost = np.sum(np.square(X_i-y_target))/N
		self.grads = {}
		dX = 2*(scores-y_target)/N
		for i in xrange(L+1,0,-1):
			indexW = 'W' + str(i-1)
			indexb = 'b' + str(i-1)
			x_old,x_in = backup[i]
			dX = dtanh(dX,x_in)
			self.grads[indexW], self.grads[indexb], dX = dlinear(dX,params[indexW],x_old) 

		return cost

	
			
		