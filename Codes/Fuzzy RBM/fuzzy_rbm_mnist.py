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

def sigmoid(X):
	return 1/(1+np.exp(-X))

x = pd.read_csv('mnist_train.csv').values[:,1:]
x = (x - np.min(x, 0)) / (np.max(x, 0) + 0.0001)  # 0-1 scaling


plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(x))
plt.show()

# Let us define the dimensions

D=784 #Visible layer
M=500 #Hidden Layer

x=x.T 

#Initialize Parameters

x_L=x
x_R=x

W_L= abs(np.random.randn(500,784))*0.01
W_R=W_L+abs(np.random.randn(500,784))*0.01 #W_R>W_L

c_L=abs(np.random.randn(500,1))*0.01
c_R=c_L+abs(np.random.randn(500,1))*0.01

b_L=abs(np.random.randn(784,1))*0.01
b_R=b_L+abs(np.random.randn(784,1))*0.01

learning_rate=0.1

for i in range(10):

	np.warnings.filterwarnings('ignore')

	p_L_h=sigmoid(c_L+np.dot(W_L,x_L))
	p_R_h=sigmoid(c_R+np.dot(W_R,x_R))
	h_L=np.where(p_L_h>=0.5,1,0)
	h_R=np.where(p_R_h>=0.5,1,0)

	temp_pLh=p_L_h
	temp_pRh=p_R_h
	temp_xL=x_L
	temp_xR=x_R

	p_L_x=sigmoid(b_L+np.dot(W_L.T,h_L))
	p_R_x=sigmoid(b_R+np.dot(W_R.T,h_R))
	x_L=np.where(p_L_x>=0.5,1,0)
	x_R=np.where(p_R_x>=0.5,1,0)

	p_L_h=sigmoid(c_L+np.dot(W_L,x_L))
	p_R_h=sigmoid(c_R+np.dot(W_R,x_R))
	h_L=np.where(p_L_h>=0.5,1,0)
	h_R=np.where(p_R_h>=0.5,1,0)

	W_L=W_L+learning_rate*(np.dot(temp_pLh,temp_xL.T) - np.dot(p_L_h,x_L.T))
	b_L=b_L+learning_rate*(temp_xL-x_L)
	c_L=c_L+learning_rate*(temp_pLh-p_L_h)

	W_R=W_R+learning_rate*(np.dot(temp_pRh,temp_xR.T) - np.dot(p_R_h,x_R.T))
	b_R=b_R+learning_rate*(temp_xR-x_R)
	c_R=c_R+learning_rate*(temp_pRh-p_R_h)


# Recovering sample (unsupervised Learning)

x_L=b_L+np.dot(W_L.T,h_L)
x_R=b_R+np.dot(W_R.T,h_R)

x_recover=(x_L+x_R)/2
x_recover=x_recover.T

plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(x_recover))
plt.show()






