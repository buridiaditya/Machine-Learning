import numpy as np
import matplotlib.pyplot as pl
import csv
import pandas as pd
import math

# Linear Regression Model
class LinearRegression:
    def __init__(self):
        self.W = None

    def predict(self,X):
        y_predict = X.dot(self.W)
        return y_predict


    def L2Cost(self, X, y_target,reg):
        cost = 0.0
        N,M = X.shape
        grads = np.ones(self.W.shape)
        scores = X.dot(self.W)
        cost = np.sum(np.square(scores - y_target))/N + reg * np.sum(self.W*self.W)
        grads = 2*X.T.dot(scores - y_target)/N + 2*self.W*reg
        return cost, grads

    def train(self,X_train, y_target,X_test,y_test, epochs=1000, learning_rate=0.05,lr_decay = 1,reg = 0.0):
        N,M = X_train.shape
        self.W = np.random.randn(M)
        cost = 1
        old_cost = 0.0
        for i in range(epochs):
            cost, dW = self.L2Cost(X_train,y_target,reg)
            self.W = self.W - learning_rate*dW
            #print("Cost after %d epochs : %f" %(i,cost))
            if(math.fabs(old_cost - cost) < 0.01):
                break;
            if i%100 == 0:
                learning_rate *= lr_decay
                #print("\nAccuracy after %d epochs : %f\n" %(i,np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N)) )
                print("Cost difference after %d epochs : %f" %(i,np.abs(cost-old_cost) ))
            old_cost = cost

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

# Data Preprocessing
# Zero centering and  Normalizing data
X_mean = np.mean(X_train,axis=0)
X_max = np.max(X_train,axis=0)
X_min = np.min(X_train,axis=0)
X_train -= X_mean
X_std = np.std(X,axis=0)
#X /= (X_max-X_min)
X_train /=X_std
X_test -= X_mean
X_test /= X_std

# append a column of ones to X
N,M = X_train.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_train
X_train = X_temp

N,M = X_test.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_test
X_test = X_temp

model = LinearRegression()

j = 0.0
rmse = []
reg = []
models = []
for i in range(20):
    reg.append(j)
    model.train(X_train,y_train,X_test,y_test,reg=j)
    rmse.append(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N))
    print("With Regularisation Value : %f The model values are : "%(j), model.W)
    j += 0.05

pl.plot(reg,rmse)
pl.xlabel("Regularization values")
pl.ylabel("RMSE values")
pl.title("Regularisation versus RMSE")
pl.show()

#cost_data = np.array(model.train(X_train,y_train,X_test,y_test))
#print(model.W)
#print("\nTest RMSE without regularization : %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )
#cost_data_reg1 = np.array(model.train(X_train,y_train,X_test,y_test,reg=0.1))
#print(model.W)
#print("\nTest RMSE with regularization 0.1: %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )
#cost_data_reg2 = np.array(model.train(X_train,y_train,X_test,y_test,reg=0.2))
#print(model.W)
#print("\nTest RMSE with regularization 0.2: %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )
#cost_data_reg3 = np.array(model.train(X_train,y_train,X_test,y_test,reg=0.3))
#print(model.W)
#print("\nTest RMSE with regularization 0.3: %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )

#axes = pl.gca()
#axes.set_ylim(100000,100000000000)
#pl.plot(range(cost_data.shape[0]),cost_data,'-',label="No Reg")
#pl.plot(range(cost_data_reg1.shape[0]),cost_data_reg1,'-',label="Reg 0.1")
#pl.plot(range(cost_data_reg2.shape[0]),cost_data_reg2,'-',label="Reg 0.2")
#pl.plot(range(cost_data_reg3.shape[0]),cost_data_reg3,'-',label="Reg 0.3")
#pl.title("cost vs epochs")
#pl.xlabel("epochs x 50")
#pl.ylabel("cost")
#pl.legend()

#pl.show()
