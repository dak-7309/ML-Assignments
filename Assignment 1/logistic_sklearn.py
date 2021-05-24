from scratch import MyPreProcessor
from sklearn import metrics 
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

preprocessor = MyPreProcessor()
[X,y] = preprocessor.pre_process(2)
[Xtrain,Xvalidation,Xtest,ytrain,yvalidation,ytest]=preprocessor.data_split_7_1_2(X,y)


logistic_regression = LogisticRegression()
# Xtrain,Xtest,ytrain,ytest=train_test_split(Xtrain, ytrain, random_state=4, test_size = 0.2, train_size=0.7)

logistic_regression.fit(Xtrain,ytrain)
ypred_test=logistic_regression.predict(Xtest)
ypred_train=logistic_regression.predict(Xtrain)

print("Accuracy for Training Set: ", metrics.accuracy_score(ytrain, ypred_train)*100)
print("Accuracy for Test Set: ", metrics.accuracy_score(ytest, ypred_test)*100)
# _____________________________________________________________________________________________



logistic_regression_sgd = SGDClassifier(loss='log',learning_rate='constant',max_iter=1000,eta0=0.05)
logistic_regression_sgd.fit(Xtrain, ytrain)
ypred_test = logistic_regression_sgd.predict(Xtest)
ypred_train = logistic_regression_sgd.predict(Xtrain)

print("Accuracy for Training Set: ", metrics.accuracy_score(ytrain, ypred_train)*100)
print("Accuracy for Test Set: ", metrics.accuracy_score(ytest, ypred_test)*100)