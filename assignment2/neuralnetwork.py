class NeuralNetworks(object):
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
		

	def loss(self,X,y_target):
		for i in 