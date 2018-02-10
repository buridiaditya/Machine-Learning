import numpy as np
import matplotlib.pyplot as py
import csv
import pandas as pd

# Linear Regression Model
class LinearRegression:
    def __init__(self):
        self.W = None

    def predict(self,X):
        y_predict = X.dot(self.W)
        return y_predict


    def L2Cost(self, X, y_target,reg):
        cost = 0.0
        M,N = X.shape
        grads = np.array(self.W.shape)
        scores = X.dot(self.W)
        cost = np.sum(np.square(scores - y_target))/N + reg * np.sum(self.W*self.W)
        grads = 2*X.T.dot(scores - y_target)/N + 2*self.W*reg
        return cost, grads

    def train(self,X_train, y_target,X_test,y_test, epochs=200, learning_rate=1e-4,lr_decay = 0.95,reg = 0.0,batch_size=None):

        N,M = X_train.shape
        self.W = np.random.randn(M)*0.01

        if batch_size is None:
            batch_size = N

        old_cost = 0.0
        for i in range(epochs):
            # To prevent overfitting of data
            if batch_size is None:
                X_batch = X_train
                y_batch = y_target
            else:
                index = np.random.choice(N,batch_size)
                X_batch = X_train[index]
                y_batch = y_target[index]
            no_of_iterations_per_epoch = int(N/batch_size)
            cost=None
            for j in range(no_of_iterations_per_epoch):
                cost, dW = self.L2Cost(X_batch,y_batch,reg)
                self.W = self.W - learning_rate*dW
            print("Cost after %d epochs : %f" %(i,cost))
            print("Cost difference %f" %(np.abs(cost-old_cost)) )
            old_cost = cost
            if i%10 == 0:
                learning_rate *= lr_decay
            print("\nAccuracy after %d epochs : %f\n" %(i,np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N)) )


# Load Data and create Training and Test data
#
filename = 'kc_house_data.csv'
data = pd.read_csv(filename)
X_y = data.as_matrix().astype('float')
N,M = X_y.shape
X = X_y[:,0:M-1]
y_target = X_y[:,M-1]
print("Shape of X : ",X.shape)
print("Shape of y_target : ",y_target.shape)

# Data Preprocessing
# Zero centering and  Normalizing data
X_mean = np.mean(X,axis=0)
X_max = np.max(X,axis=0)
X_min = np.min(X,axis=0)
X -= X_mean
X_std = np.std(X,axis=0)
#X /= (X_max-X_min)
X/=X_std

# append a column of ones to X
N,M = X.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X
X = X_temp

# Split data into train and test data
#
size_of_train = int(N*0.8)
X_train = X[0:size_of_train]
X_test = X[size_of_train:N]
y_train = y_target[0:size_of_train]
y_test = y_target[size_of_train:N]
print("Shape of X_train : ", X_train.shape)
print("Shape of X_test : ", X_test.shape)
print("Shape of y_train : ", y_train.shape)
print("Shape of y_test : ", y_test.shape)

model = LinearRegression()

model.train(X_train,y_train,X_test,y_test)
