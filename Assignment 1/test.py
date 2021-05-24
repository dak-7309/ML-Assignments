from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
import pandas as pd


preprocessor = MyPreProcessor()
print('Linear Regression')

# Xtrain = np.array([[1,2,3], [4,5,6]])
# ytrain = np.array([1,2])
# [Xtrain,ytrain]= preprocessor.pre_process(Xtrain,ytrain,-1)

# Xtest = np.array([[7,8,9],[10,11,12],[13,14,15]])
# [Xtest,ytest] = preprocessor.pre_process(Xtest,[],-1)
# ytest = np.array([3,4,5])

# linear = MyLinearRegression(0.0005)
# linear.fit(Xtrain, ytrain)
# ypred = linear.predict(Xtest)
# print('Predicted Values:', ypred)
# print('True Values:', ytest)
# # # ______________________________________________________________________________________________



# DATASET 1
[Xtrain,ytrain]= preprocessor.pre_process( 0)
# alpha=0.005,iter=1500
linear = MyLinearRegression(0.005,2000)
# compare rmse vs mae for kfold
linear.RMSE_vs_MAE(Xtrain,ytrain,10)
# plot training loss vs iterations, validation loss vs iterations (from the kfold) using rmse
# K=5 is default value, 0 flag means rmse, 1 means mae
linear.plot_training_vs_validation(Xtrain,ytrain,10,1)
# use normal form to find optimal parameters and use it to find training loss and validation loss for best fold, using MAE which is the better loss
linear.Normal_form(Xtrain,ytrain,10)

# DATASET 2
[Xtrain,ytrain]= preprocessor.pre_process(1)
# alpha=0.0002,iter=500
linear = MyLinearRegression(0.0002,600)
# compare rmse vs mae for kfold
linear.RMSE_vs_MAE(Xtrain,ytrain,10)
# plot training loss vs iterations, validation loss vs iterations (from the kfold) using rmse
# K=5 is default value, 0 flag means rmse, 1 means mae
linear.plot_training_vs_validation(Xtrain,ytrain,10,1)


# _____________________________________________________________________________________________________


print('Logistic Regression')
# preprocess data for Dataset 2
[X,y]= preprocessor.pre_process(2)
# obtain training, vaildation an test sets, 7:1:2
[Xtrain,Xvalidation,Xtest,ytrain,yvalidation,ytest]=preprocessor.data_split_7_1_2(X,y)


# # BATCH GRADIENT DESCENT
# alpha=0.01, iter=500
print("Batch GD- alpha=0.05 iter=1000")
logistic=MyLogisticRegression(0.05,1000,Xvalidation,yvalidation)
logistic.fit(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set

# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)

# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()


# ________________________________________________________________________________________________


# alpha=0.01, iter=500
print("Batch GD- alpha=0.01, iter=1000")
logistic=MyLogisticRegression(0.01,1000,Xvalidation,yvalidation)
logistic.fit(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)

print()


# ________________________________________________________________________________________________


# # alpha=0.0001,iter=10000
print("Batch GD- alpha=0.0001, iter=15000")
logistic=MyLogisticRegression(0.0001,15000,Xvalidation,yvalidation)
logistic.fit(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()

 # ______________________________________________________________________________________________


# # alpha=10,iter=100
print("Batch GD- alpha=10 iter=500")
logistic=MyLogisticRegression(10,500,Xvalidation,yvalidation)
logistic.fit(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()

# ______________________________________________________________________________________________


# # STOCHASTIC GRADIENT DESCENT
# # alpha=0.05,iter=500
print("Stochastic GD- alpha=0.05 iter=1000")
logistic=MyLogisticRegression(0.05,1000,Xvalidation,yvalidation)
logistic.fit_stochastic(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()


# # # ________________________________________________________________________________________________





# # alpha=0.01,iter=500
print("Stochastic GD- alpha=0.01 iter=1000")
logistic=MyLogisticRegression(0.01,1000,Xvalidation,yvalidation)
logistic.fit_stochastic(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()

# # # ________________________________________________________________________________________________

# # alpha=0.0001
print("Stochastic GD- alpha=0.0001 iter=15000")
logistic=MyLogisticRegression(0.0001,15000,Xvalidation,yvalidation)
logistic.fit_stochastic(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()

# # # ______________________________________________________________________________________________


# # alpha=10
print("Stochastic GD- alpha=10 iter=500")
logistic=MyLogisticRegression(10,500,Xvalidation,yvalidation)
logistic.fit_stochastic(Xtrain,ytrain)

training_error_vs_iterations=logistic.J_history 
validation_error_vs_iterations=logistic.J_history_valid

# calculates accuracy for Training Set
YY=logistic.predict(Xtrain)
print("Training set")
logistic.cal_accuracy(YY,ytrain)
# training loss vs iterations


# calculates accuracy for Validation Set
# YY=logistic.predict(Xvalidation)
# logistic.cal_accuracy(YY,yvalidation)
# validation loss vs iterations
# validation_error_vs_iterations=logistic.J_history -------------------------

# plots training loss vs iterations, validation loss vs iterations
logistic.plot_training_vs_validations_iterations(training_error_vs_iterations,validation_error_vs_iterations)


# calculates accuracy for Test Set
YY=logistic.predict(Xtest)
print("Testing set")
logistic.cal_accuracy(YY,ytest)
print()

# # # ______________________________________________________________________________________________

