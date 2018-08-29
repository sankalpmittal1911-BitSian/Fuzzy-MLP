import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
 
plt.rcParams['image.cmap'] = 'gray'
import math
import csv
import pandas as pd
np.warnings.filterwarnings('ignore')
 
def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)
 
def relu(X):
    X[X<=0]=0
    return X
 
def sigmoid(X):
    return 1/(1+np.exp(-X))
 
x = pd.read_csv('mnist_train.csv').values[:,1:]
x = (x - np.min(x, 0)) / (np.max(x, 0) + 0.0001)  # 0-1 scaling
 
 
plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(x))
plt.show()
 
 
x=x.T
 
#Initialize Parameters
 
N=5999
 
W1_L= np.random.randn(500,784)*0.01
W1_R=W1_L+abs(np.random.randn(500,784))*0.01 #W_R>W_L
 
b1_L=np.random.randn(500,1)*0.01
b1_R=b1_L+abs(np.random.randn(500,1))*0.01
 
b2_L=np.random.randn(784,1)*0.01
b2_R=b2_L+abs(np.random.randn(784,1))*0.01
 
W2_L= np.random.randn(784,500)*0.01
W2_R=W2_L+abs(np.random.randn(784,500))*0.01 #W_R>W_L
 
learning_rate=0.1
 
for i in range(200):
 
    np.warnings.filterwarnings('ignore')
 
    #Forward Propagation
 
    a1L=relu(np.dot(W1_L,x)+b1_L)
    a1R=relu(np.dot(W1_R,x)+b1_R)
    a2L=sigmoid(np.dot(W2_L,a1L)+b2_L)
    a2R=sigmoid(np.dot(W2_R,a1R)+b2_R)
 
    #Calculations of gradients
 
    dW1_L=np.dot(np.dot(W2_L.T,x),((a2L-x)*(1-a2L)*a2L).T) *(1/N)
    dW1_R=np.dot(np.dot(W2_R.T,x),((a2R-x)*(1-a2R)*a2R).T) *(1/N)
 
    db1_L=np.sum(np.dot(W2_L.T,(a2L-x)*(1-a2L)*a2L),axis=0,keepdims=True) *(1/N)
    db1_R=np.sum(np.dot(W2_R.T,(a2R-x)*(1-a2R)*a2R),axis=0,keepdims=True) *(1/N)
 
    dW2_L=np.dot((a2L-x)*(1-a2L)*a2L,a1L.T)*(1/N)
    dW2_R=np.dot((a2R-x)*(1-a2R)*a2R,a1R.T)*(1/N)
 
    db2_L=np.sum((a2L-x)*a2L*(1-a2L),axis=0)*(1/N)
    db2_R=np.sum((a2R-x)*a2R*(1-a2R),axis=0)*(1/N)
 
    #Backward Propagation
 
    W1_L=W1_L - learning_rate*dW1_L
    W1_R=W1_R - learning_rate*dW1_R
 
    b1_L=b1_L - learning_rate*db1_L
    b1_R=b1_R - learning_rate*db1_R
 
    W2_L=W2_L - learning_rate*dW2_L
    W2_R=W2_R - learning_rate*dW2_R
 
    b2_L=b2_L - learning_rate*db2_L
    b2_R=b2_R - learning_rate*db2_R
 
 
 
# Recovering sample (supervised Learning)
 
a1L=relu(np.dot(W1_L,x)+b1_L)
a1R=relu(np.dot(W1_R,x)+b1_R)
a2L=sigmoid(np.dot(W2_L,a1L)+b2_L)
a2R=sigmoid(np.dot(W2_R,a1R)+b2_R)
 
x_recover=(a2L+a2R)/2
x_recover=x_recover.T
 
plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(x_recover))
plt.show()
