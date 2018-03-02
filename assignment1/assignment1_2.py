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

    def trainSGD(self,X_train, y_target,X_test,y_test, epochs=1000, learning_rate=0.05,lr_decay = 1,reg = 0.0,batch_size=None):
        N,M = X_train.shape
        N_test,_ = X_test.shape
        self.W = np.random.randn(M)
        old_cost = 0.0
        cost_data = []
        epochs_list = []
        for i in range(epochs):
            cost, dW = self.L2Cost(X_train,y_target,reg)
            self.W = self.W - learning_rate*dW
            if i%10 == 0:
                epochs_list.append(i)
                cost_data.append(np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N_test))
            if((math.fabs(old_cost - cost) < 0.01)):
                break;
            if i%100 == 0:
                learning_rate *= lr_decay
                print("Cost difference after %d epochs : %f" %(i,np.abs(cost-old_cost) ))
                #print("\nTest Error after %d epochs : %f\n" %(i,np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N_test)) )
            old_cost = cost
        return epochs_list,cost_data

    def trainIRLS(self,X_train, y_target,X_test,y_test,epochs = 1000, reg = 0.0):
        old_cost = 0.0
        N,M = X_train.shape
        N_test,_ = X_test.shape
        cost_data = []
        epochs_list = []
        self.W = np.random.randn(M)
        #for i in range(epochs):
        epochs_list.append(0)
        cost_data.append(np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N_test))

        cost, dW = self.L2Cost(X_train,y_train,reg)
        H = 2*X_train.T.dot(X_train)/N
        H_inv = np.linalg.inv(H)
        self.W = self.W - H_inv.dot(dW)
           # if i%100 == 0:
        epochs_list.append(1)
        cost_data.append(np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N_test))
           # if(math.fabs(old_cost-cost) < 0.01):
           #     break;
           # if i%1000 == 0:
           #     print("Cost after %d epochs %f"%(i,cost))
           # old_cost = cost
        print("\nTest Error using IRLS : %f\n" %(np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N_test)) )
        return epochs_list,cost_data
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

epochs_list_sgd = []
epochs_list_irls = []
rmse_sgd = []
rmse_irls = []
epochs_list_sgd,rmse_sgd = model.trainSGD(X_train,y_train,X_test,y_test,epochs=10000)
print("\nTest Error using Gradient Descent : %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )
print("Gradient Descent trained model parameters : ",model.W)
epochs_list_irls,rmse_irls = model.trainIRLS(X_train,y_train,X_test,y_test,epochs=10000)
print("IRLS trained model parameters : ",model.W)

pl.plot(epochs_list_sgd,rmse_sgd,'-',label="Gradient Descent")
pl.plot(epochs_list_irls,rmse_irls,'-',label="IRLS")
pl.xlabel("epochs")
pl.ylabel("RMSE")
pl.title("RMSE-Performance of Various learning algorithms vs iterations")
pl.legend()
pl.show()

"""
model.trainSGD(X_train,y_train,X_test,y_test)
print("\nRMSE with SGD: %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )
model.trainIRLS(X_train,y_train,X_test,y_test)
print("\nRMSE with IRLS : %f\n" %(np.sqrt(np.sum(np.square(model.predict(X_test)-y_test))/N)) )
"""
