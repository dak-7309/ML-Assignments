import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import *
from math import *
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scratch import MyLogisticRegression, MyPreProcessor
preprocessor = MyPreProcessor()

X=[]
y=[]
# read q4 dataset
df=pd.read_csv('q4.csv',names=["ylabel","x1","x2"])

t1=np.array(df["ylabel"].values)
t2=np.array(df["x1"].values)
t3=np.array(df["x2"].values)


tt=np.array([])
for i in range(len(t1)):
    Z=np.array([t1[i],t2[i],t3[i]])
    tt=np.append(tt,Z)
tt=tt.reshape(len(tt)//3,3)

for i in range(len(tt)):
    Z=np.array( [tt[i][1],tt[i][2]] )
    X=np.append(X,Z)
    Z=np.array([tt[i][0]])
    y=np.append(y,Z)

# reshaping x and y matrices
X=X.reshape(len(X)//2,2)
print(X)
# append column of 1s
# X=np.insert(X, 0, [1], axis=1)
# y=y.reshape(-1,1)


# Xtrain=np.array([])
# ytrain=np.array([])
# Xvalidation=np.array([])
# yvalidation=np.array([])
# Xtest=np.array([])
# ytest=np.array([])

# val=len(X)//10
# for i in range(len(X)):
#     if i<7*val:
#         Xtrain=np.append(Xtrain,X[i])
#         ytrain=np.append(ytrain,y[i])
#     elif i>=7*val and i<8*val:
#         Xvalidation=np.append(Xvalidation,X[i])
#         yvalidation=np.append(yvalidation,y[i])
#     else:
#         Xtest=np.append(Xtest,X[i])
#         ytest=np.append(ytest,y[i])

# # reshaping x and y matrices for the 3 sets
# Xtrain=Xtrain.reshape(len(Xtrain)//3,3)
# Xvalidation=Xvalidation.reshape(len(Xvalidation)//3,3)
# Xtest=Xtest.reshape(len(Xtest)//3,3)
# ytrain=ytrain.reshape(-1,1)
# yvalidation=yvalidation.reshape(-1,1)
# ytest=ytest.reshape(-1,1)

# print(Xtrain)
# print(ytrain)
# # resizing matrices

# # fit my model for this dataset and print Thetas

# # logistic=MyLogisticRegression(0.005,10000,Xvalidation,yvalidation)
logistic = LogisticRegression()
logistic.fit(X,y)

# logistic.fit(Xtrain,ytrain)
print(logistic.coef_)
print(logistic.intercept_)




# # calculates accuracy for Training Set
# YY=logistic.predict(Xtrain)
# print("Training set")
# logistic.cal_accuracy(YY,ytrain)
# # training loss vs iterations


# # calculates accuracy for Test Set
# YY=logistic.predict(Xtest)
# print("Testing set")
# logistic.cal_accuracy(YY,ytest)
# print()


