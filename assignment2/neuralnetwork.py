import numpy as np

class NeuralNetwork(object):
	def __init__(self,input_dim,hidden_dims,num_classes,activation_function="tanh",weight_scale=10e-2):
		L = hidden_dims.shape[0]
		self.params = {}
		self.activation = activation_function
		self.hidden_layers = L
		for i in range(L+1):
			indexW = 'W' + str(i)
			indexb = 'b' + str(i)
			if i == 0:
				self.params[indexW] = np.random.normal(0,weight_scale,(input_dim,hidden_dims[i]))
				self.params[indexb] = np.random.normal(0,weight_scale,(hidden_dims[i]))
			elif i == L:
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

		# Forward Pass
		for i in range(self.hidden_layers + 1):
			indexW = 'W' + str(i)
			indexb = 'b' + str(i)	
			W = self.params[indexW] 
			b = self.params[indexb] 
			X_old = X_i
			
			X_i = self.linear(X_i,W,b)
			# Use any activation fucntion
			if(self.activation == "sigmoid"):
				X_i = self.act_sigmoid(X_i)
			elif(self.activation == "tanh"):
				X_i = self.act_tanh(X_i)
			backup.append((X_old,X_i))

		# For tanh activation function
		if(self.activation == "tanh"):
			y_target[y_target == 0] = -1
		scores = X_i

		# L2 error
		cost = np.sum(np.square(X_i-y_target))/N
		self.grads = {}
		dX = 2*(scores-y_target)/N

		# Backward Pass
		for i in range(self.hidden_layers+1,0,-1):			
			indexW = 'W' + str(i-1)
			indexb = 'b' + str(i-1)
			x_old,x_in = backup[i-1]
			if(self.activation == "sigmoid"):
				dX = self.dsigmoid(dX,x_in)
			elif(self.activation == "tanh"):
				dX = self.dtanh(dX,x_in)
			
			self.grads[indexW], self.grads[indexb], dX = self.dlinear(dX,self.params[indexW],x_old) 

		return cost

	def train(self,X_train,y_train,X_test,y_test,epochs=1000,learning_rate=10e-4,batch_size=100):
		N, M = X_train.shape
		no_of_batches = int(N/batch_size)
		
		X_s = X_train[0:no_of_batches*batch_size]
		X_s_t = X_train[no_of_batches*batch_size:]
		X_batches = np.split(X_s,no_of_batches)
		X_batches.append(X_s_t)

		y_s = y_train[0:no_of_batches*batch_size]
		y_s_t = y_train[no_of_batches*batch_size:]
		y_batches = np.split(y_s,no_of_batches)
		y_batches.append(y_s_t)

		for i in range(epochs):
			for j in range(len(X_batches)):				
				cost = self.loss_1(X_batches[j],y_batches[j])
				print (cost)
				for i in range(self.hidden_layers + 1):
					indexW = 'W' + str(i)
					indexb = 'b' + str(i)	
					W = self.params[indexW] 
					b = self.params[indexb] 
		