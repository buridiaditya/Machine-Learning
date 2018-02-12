import numpy as np
import matplotlib.pyplot as pl
import csv
import math
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
        N,M = X.shape
        grads = np.zeros(self.W.shape)
        scores = X.dot(self.W)
        cost = np.sum(np.square(scores - y_target))/N + reg * np.sum(self.W*self.W)
        grads = 2*X.T.dot(scores - y_target)/N + 2*self.W*reg
        return cost, grads

    def train(self,X_train, y_target,X_test,y_test, epochs=20000, learning_rate=0.05,lr_decay = 1,reg = 0.0):
        N,M = X_train.shape
        self.W = np.random.randn(M)
        old_cost = 0.0
        cost_data = []
        for i in range(epochs):
            cost, dW = self.L2Cost(X_train,y_target,reg)
            self.W = self.W - learning_rate*dW

            #print("Cost after %d epochs : %f" %(i,cost))
            if i%100 == 0:
                learning_rate *= lr_decay
                #print("\nAccuracy after %d epochs : %f\n" %(i,np.sqrt(np.sum(np.square(self.predict(X_test)-y_test))/N)) )
                print("Cost difference after %d epochs : %f" %(i,np.abs(cost-old_cost)) )
            if i%1000 == 0:
                cost_data.append(cost)
            old_cost = cost

        return cost_data

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

# Adds quadratic cubic features
N,M = X.shape
X_temp = np.ones((N,3*M))
X_temp[:,0:M] = X
X_l = X_temp[:,0:M]
X_temp[:,M:2*M] = X*X
X_qa = X_temp[:,0:2*M]
X_temp[:,2*M:3*M] = X*X*X
X_cu = X_temp

# Split data into train and test data
#
size_of_train = int(N*0.8)
X_train_l = X_l[0:size_of_train]
X_test_l = X_l[size_of_train:N]
y_train = y_target[0:size_of_train]
y_test = y_target[size_of_train:N]
print("Shape of X_train : ", X_train_l.shape)
print("Shape of X_test : ", X_test_l.shape)
print("Shape of y_train : ", y_train.shape)
print("Shape of y_test : ", y_test.shape)

X_train_qa = X_qa[0:size_of_train]
X_test_qa = X_qa[size_of_train:N]
print("Shape of X_train_qa : ", X_train_qa.shape)
print("Shape of X_test_qa : ", X_test_qa.shape)

X_train_cu = X_cu[0:size_of_train]
X_test_cu = X_cu[size_of_train:N]
print("Shape of X_train_cu : ", X_train_cu.shape)
print("Shape of X_test_cu : ", X_test_cu.shape)

# Data Preprocessing
# Zero centering and  Normalizing data
X_mean_l = np.mean(X_train_l,axis=0)
X_max_l = np.max(X_train_l,axis=0)
X_min_l = np.min(X_train_l,axis=0)
X_std_l = np.std(X_train_l,axis=0)

X_train_l -= X_mean_l
#X_train_l /= (X_max_l-X_min_l)
X_train_l /= X_std_l
X_test_l -= X_mean_l
X_test_l /= X_std_l
#X_test_l /= (X_max_l-X_min_l)

X_mean_qa = np.mean(X_train_qa,axis=0)
X_max_qa = np.max(X_train_qa,axis=0)
X_min_qa = np.min(X_train_qa,axis=0)
X_std_qa = np.std(X_train_qa,axis=0)

X_train_qa -= X_mean_qa
#X_train_qa /= (X_max_qa-X_min_qa)
X_train_qa /= X_std_qa
X_test_qa -= X_mean_qa
X_test_qa /= X_std_qa
#X_test_qa /= (X_max_qa-X_min_qa)

X_mean_cu = np.mean(X_train_cu,axis=0)
X_max_cu = np.max(X_train_cu,axis=0)
X_min_cu = np.min(X_train_cu,axis=0)
X_std_cu = np.std(X_train_cu,axis=0)

X_train_cu -= X_mean_cu
#X_train_cu /= (X_max_cu-X_min_cu)
X_train_cu /= X_std_cu
X_test_cu -= X_mean_cu
X_test_cu /= X_std_cu
#X_test_cu /= (X_max_cu-X_min_cu)



# Append column of ones to X
N,M = X_train_l.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_train_l
X_train_l = X_temp

N,M = X_test_l.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_test_l
X_test_l = X_temp


N,M = X_train_qa.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_train_qa
X_train_qa = X_temp

N,M = X_test_qa.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_test_qa
X_test_qa = X_temp


N,M = X_train_cu.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_train_cu
X_train_cu = X_temp

N,M = X_test_cu.shape
X_temp = np.ones((N,M+1))
M += 1
X_temp[:,1:M] = X_test_cu
X_test_cu = X_temp


model = LinearRegression()

cost_data = np.array(model.train(X_train_l,y_train,X_test_l,y_test))
cost_data_sq = np.array(model.train(X_train_qa,y_train,X_test_qa,y_test))
cost_data_cu = np.array(model.train(X_train_cu,y_train,X_test_cu,y_test))

axes = pl.gca()
#axes.set_ylim(0,1000)
pl.plot(range(cost_data.shape[0]),cost_data,'-',label="Linear")
pl.plot(range(cost_data_sq.shape[0]),cost_data_sq,'-',label="Square")
pl.plot(range(cost_data_cu.shape[0]),cost_data_cu,'-',label="Cubic")

pl.title("cost vs epochs")
pl.legend()

pl.show()


