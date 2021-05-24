# -*- coding: utf-8 -*-
"""MLq2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X9oJLb997RTqhFd-6UVasAgTQr3vgPsf
"""

# Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# drive.mount('/content/drive/')
# %cd /content/drive/MyDrive/ML_ass3

from Q1 import MyNeuralNetwork
from sklearn.utils import shuffle
from math import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neural_network import MLPClassifier

scaler=StandardScaler()
X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
# importing MNIST dataset
y=y.astype(int)

# shuffling dataset, both x and ys corresponding to each other
random_state=check_random_state(0)
permutation=random_state.permutation(X.shape[0])
X=X[permutation]
y=y[permutation]
X=X.reshape((X.shape[0],-1))

# splitting data into train test :: 80:20
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# standardising and scaling data using StandardScaler
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
xxttrr,xxttee,yyttrr,yyttee=X_train,X_test,y_train,y_test
# storing

# one hot encoding y that helps in further steps, both ytrain and ytest
encoder_=OneHotEncoder(sparse=False,categories='auto')
y_train=encoder_.fit_transform(y_train.reshape(len(y_train),-1))
y_test=encoder_.transform(y_test.reshape(len(y_test),-1))

l=[784,256,128,64,10]
# initialising my neural network for the layers provided in the question
# running my network individually for various activation fns 1 by 1 and training model to determine testing and training accuracy
nn=MyNeuralNetwork(n_layers=len(l),layers_sizes=l,activation="tanh",weight_init="normal",batch_size=200)
nn.XTEST=X_test
nn.YTEST=y_test
nn.fit(X_train,y_train)

# currently commented, saves model weights for all my classifiers
# f = open("/content/drive/MyDrive/ML ass3/weights_linear.pkl", "wb")
# pickle.dump(nn.weights,f)
# f.close()

# plotting testing and training accuracy together
print("Training Accuracy:",nn.score(X_train,y_train))
print("Testing Accuracy:",nn.score(X_test,y_test))
nn.plot_cost()

nn.TTTT
nn.TTTT.shape
ARRAY=np.transpose(nn.TTTT)
ARRAY.shape
# my output after the 2nd last layer of activation, I saved it inside a class object attribute
# in the forward propagation step, need it to perform tSNE as required

# svd is used to perform dimensionality reduction
def SVD(X):
    svd = TruncatedSVD(n_components=20,random_state=0)
    X_svd=svd.fit(X).transform(X)
    return X_svd
# used to visualize high dimensional data in 2 components using scatterplots, 10 class labels here
def tSNE(X,Y):
    tsn=TSNE(n_components=2,random_state=0)
    X_tsn=tsn.fit_transform(X)
    sns.scatterplot(X_tsn[:,0], X_tsn[:,1], hue=Y, legend='full',palette=sns.color_palette("hls", 10))
    plt.show()

X_svd=SVD(ARRAY)
tSNE(X_svd,yyttee)

# comparing my models with Sklearn's implementation for all 4 activation fns
# sigmoid,linear,tanh,relu

clf = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="logistic", batch_size=200, learning_rate_init=0.1, max_iter=100, random_state=1, solver="sgd")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="identity", batch_size=200, learning_rate_init=0.1, max_iter=100, random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

"""83.789"""

clf = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="tanh", batch_size=200, learning_rate_init=0.1, max_iter=100, random_state=1, solver="sgd")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", batch_size=200, learning_rate_init=0.1, max_iter=100, random_state=1, solver="sgd")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))