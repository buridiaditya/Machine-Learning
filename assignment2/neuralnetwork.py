import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
	def __init__(self,input_dim,hidden_dims,num_classes,activation_function="tanh",weight_scale=10e-2):
		L = hidden_dims.shape[0]
		self.params = {}
		# Store which activation function to use
		self.activation = activation_function
		self.hidden_layers = L
		# Create a map of  weight and bias for every layer
		# Initialize all the weights randomly sampled from normal distribution

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

	def loss_1(self,X,y_target,predict=False):
		N  = X.shape[0]
		X_i = X
		backup = []
		self.grads = {}
		# Forward Pass
		for i in range(self.hidden_layers + 1):
			indexW = 'W' + str(i)
			indexb = 'b' + str(i)
			W = self.params[indexW]
			b = self.params[indexb]
			X_old = X_i.copy()

			# Affine function
			X_i = self.linear(X_i,W,b)
			# Activation function
			if(self.activation == "sigmoid"):
				X_i = self.act_sigmoid(X_i)
			elif(self.activation == "tanh"):
				X_i = self.act_tanh(X_i)
			# Store node data for use in backward pass
			backup.append((X_old,X_i))

		scores = X_i.copy()
		if(predict == True):
			return scores

		y_target_ = y_target.copy()
		# For tanh activation function the labels of 0 replaced with -1
		if(self.activation == "tanh"):
			y_target_[y_target_ == 0] = -1


		# L2 error
		cost = np.sum(np.square(X_i-y_target_))/N

		# Backward Pass
		dX = 2*(scores-y_target_)/N
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

	def loss_2(self,X,y_target,predict=False):
		N,_  = X.shape
		X_i = X
		backup = []
		self.grads = {}
		# Forward Pass
		for i in range(self.hidden_layers + 1):
			indexW = 'W' + str(i)
			indexb = 'b' + str(i)
			W = self.params[indexW]
			b = self.params[indexb]
			X_old = X_i.copy()

			# Affine function
			X_i = self.linear(X_i,W,b)
			# Activation function
			if(i != self.hidden_layers):
				if(self.activation == "sigmoid"):
					X_i = self.act_sigmoid(X_i)
				elif(self.activation == "tanh"):
					X_i = self.act_tanh(X_i)
			# Store node data for use in backward pass
			backup.append((X_old,X_i))

		scores = X_i.copy()
		temp = np.exp(X_i)
		scores = temp / np.reshape(np.sum(temp,axis=1),(N,1))

		scores_row1 = np.reshape(scores[:,0],(N,1)).copy()
		scores_row2 = np.reshape(scores[:,1],(N,1)).copy()

		if(predict == True):
			return scores_row1

		# L2 error
		#Softmax function
		cost = np.sum(np.square(scores_row1-y_target))/N

		# Backward Pass
		dX = 2*(scores_row1-y_target)/N
		#temp = scores
		#temp[:,[0,1]]  = scores[:,[1, 0]]

		temp = 1-scores
		temp[:,1] *= -1
		dX = dX*(scores*temp)

		for i in range(self.hidden_layers+1,0,-1):
			indexW = 'W' + str(i-1)
			indexb = 'b' + str(i-1)
			x_old,x_in = backup[i-1]
			if(i != self.hidden_layers +1):
				if(self.activation == "sigmoid"):
					dX = self.dsigmoid(dX,x_in)
				elif(self.activation == "tanh"):
					dX = self.dtanh(dX,x_in)

			self.grads[indexW], self.grads[indexb], dX = self.dlinear(dX,self.params[indexW],x_old)

		return cost

	def predict(self,X_test,y_test):
		cost = np.array([])
		if(self.method == "one"):
			cost = self.loss_1(X_test,y_test,predict=True)
		else:
			cost = self.loss_2(X_test,y_test,predict=True)

		if(self.activation == "sigmoid" or self.method == "two"):
			cost[cost > 0.5] = 1
			cost[cost <= 0.5] = 0
		else:
			cost[cost > 0] = 1
			cost[cost <= 0] = 0

		return cost

	def plot(self):
		plt.plot(self.no_epochs,self.square_error_train,'-',label='Insample Error')
		plt.plot(self.no_epochs,self.square_error_test,'-',label='Outsample Error')
		plt.xlabel("Epochs")
		plt.ylabel("Error")
		plt.title("Error vs Epochs")
		plt.legend()
		plt.show()


	def train(self,X_train,y_train,X_test,y_test,epochs=100,learning_rate=1e-3,learning_rate_decay=0.95,batch_size=1000,method="one"):
		N, M = X_train.shape
		no_of_batches = int(N/batch_size)
		#print(no_of_batches)
		X_s = X_train[0:no_of_batches*batch_size]
		X_s_t = X_train[no_of_batches*batch_size:]
		X_batches = np.split(X_s,no_of_batches)
		
		if(N%batch_size):
			X_batches.append(X_s_t)

		y_s = y_train[0:no_of_batches*batch_size]
		y_s_t = y_train[no_of_batches*batch_size:]
		y_batches = np.split(y_s,no_of_batches)

		if(N%batch_size):
			y_batches.append(y_s_t)
		
		self.no_epochs = []
		self.square_error_train = []
		self.square_error_test = []
		self.method = method
		
		cost = 0.0
		for i in range(epochs):
			cost = 0.0
			
			if(i%25==0):
				prediction = self.predict(X_test,y_test)
				print("-----------Ratio of Correct predictions over testset(%d/%d)-----------"%(np.sum(prediction == y_test),y_test.shape[0]))
				#print (prediction)
				prediction = self.predict(X_train,y_train)
				print("-----------Ratio of Correct predictions over training set(%d/%d)-----------"%(np.sum(prediction == y_train),y_train.shape[0]))

			if((i+1)%25==0):
				learning_rate *= learning_rate_decay

			for j in range(no_of_batches+ ( 1 if(N%batch_size) else 0 ) ):
				#print (X_batches[j].shape)
				temp1 = 0.0
				if(method == "two"):
					temp1 = self.loss_2(X_batches[j],y_batches[j])
				else:
					temp1 = self.loss_1(X_batches[j],y_batches[j])
				cost += temp1
				for k in range(self.hidden_layers + 1):
					indexW = 'W' + str(k)
					indexb = 'b' + str(k)
					W = self.params[indexW]
					b = self.params[indexb]
					self.params[indexW] = W - learning_rate*self.grads[indexW]
					self.params[indexb] = b - learning_rate*self.grads[indexb]

			print("Epoch (%d/%d) Training Error : %f"%(i+1,epochs,cost/( no_of_batches+ ( 1 if(N%batch_size) else 0 ) ) ) )

			prediction = self.predict(X_test,y_test)
			outsample_cost = np.sum(np.square(prediction - y_test))/y_test.shape[0]
			self.no_epochs.append(i)
			self.square_error_train.append(cost)
			self.square_error_test.append(outsample_cost)
			if(cost/( no_of_batches+ ( 1 if(N%batch_size) else 0 )) < 0.005):
				break;

		prediction = self.predict(X_test,y_test)
		#print(prediction)
		print("-----------Ratio of Correct predictions over testset(%d/%d)-----------"%(np.sum(prediction == y_test),y_test.shape[0]))
		prediction = self.predict(X_train,y_train)
		print("-----------Ratio of Correct predictions over training set(%d/%d)-----------"%(np.sum(prediction == y_train),y_train.shape[0]))
