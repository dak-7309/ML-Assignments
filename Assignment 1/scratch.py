import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import *
from math import *

class MyPreProcessor():

    def __init__(self):
        pass

    # to perform Exploratory data analysis on DataSet 3
    def eda(self,df):
        print("Pairwise correlation of features, using standard correlation coefficient")
        print(df.corr(method='pearson'))
        
        print()
        print("First 5 rows")
        print(df.head())

        print()
        print("Data Info")
        print(df.info())

        print()
        print("Presence of null values")
        print(df.isnull().sum())

        print()
        print("Data Description")
        print(df.describe())

        print()
        print("PairPlots")
        sns.pairplot(df,hue='class')
        plt.show()

        print()
        print("Class 0 vs Class 1")
        sns.countplot(x='class',data=df)
        plt.title('Classes 1 vs 0')
        plt.show()
        print(df['class'].value_counts())

        print()
        print("Attribute distribution plot")
        df.hist(figsize=(20,10), grid =False, layout = (2,4), bins=30)
        plt.show()

        print()
        print("Attribute distribution plot")
        
        sns.displot(df, x='variance', hue='class', alpha = 0.3)
        plt.show()
        sns.displot(df, x='skewness', hue='class', alpha = 0.3)
        plt.show()
        sns.displot(df, x='curtosis', hue='class', alpha = 0.3)
        plt.show()
        sns.displot(df, x='entropy', hue='class', alpha = 0.3)
        plt.show()


    def pre_process(self, dataset):
        X=[]
        y=[]
        # if input is inputted by user, np array
        if dataset==-1:
            # append column of 1s in the start
            Xtrain=np.insert(X, 0, [1], axis=1)
            if len(y)!=0:
                # transpose
                ytrain=y.reshape(-1,1)
            else:
                ytrain=y

        # abalone dataset for Linear Regression
        if dataset == 0:
            df=pd.read_csv('Dataset.data', header=None)
            l=df[0].values

            for i in range(len(l)):
                start_train=l[i].find(" ")+1
                end_train=l[i].rfind(" ")-1
                start_pred=l[i].rfind(" ")+1

                # split columns by spaces, ignore gender and split into X and y 
                A=list(map(float,l[i][start_train:end_train+1].split()))
                X=np.append(X,A)
                B=float(l[i][start_pred:])
                y=np.append(y,[B])

            # reshaping x and y matrices
            X=X.reshape(len(X)//7,7)
            Xtrain=np.insert(X, 0, [1], axis=1)
            y=y.reshape(1,len(y))[0]
            ytrain=y.reshape(-1,1)
            pass
        

        elif dataset == 1:
            df = pd.read_csv('VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv') 
            t1=np.array(df["Critic_Score"].values)
            t2=np.array(df["User_Score"].values)
            t3=np.array(df["Global_Sales"].values)

            for i in range(len(t1)):
                # removing rows with nan and "tbd" values, type cast to float
                if not np.isnan(t1[i]) and t2[i]!="tbd" and not np.isnan(float(t2[i])):
                    Z=np.array([t1[i],float(t2[i])])
                    X=np.append(X,Z)
                    Z=np.array([t3[i]])
                    y=np.append(y,Z)

            # reshaping x and y matrices
            X=X.reshape(len(X)//2,2)
            Xtrain=np.insert(X, 0, [1], axis=1)
            ytrain=y.reshape(-1,1)
            pass
            
            
        
        elif dataset == 2:
            df=pd.read_csv('dataset3.csv',names=["variance","skewness","curtosis","entropy","class"])
            # applying exploratory analysis
            self.eda(df)

            t1=np.array(df["variance"].values)
            t2=np.array(df["skewness"].values)
            t3=np.array(df["curtosis"].values)
            t4=np.array(df["entropy"].values)
            t5=np.array(df["class"].values)


            tt=np.array([])
            for i in range(len(t1)):
                Z=np.array([t1[i],t2[i],t3[i],t4[i],t5[i]])
                tt=np.append(tt,Z)
            tt=tt.reshape(len(tt)//5,5)
            np.random.seed(14) 
            np.random.shuffle(tt)
            # shuffle up rows, fixing up seed value for same shuffle

            for i in range(len(tt)):
                Z=np.array( [tt[i][0],tt[i][1],tt[i][2],tt[i][3]] )
                X=np.append(X,Z)
                Z=np.array([tt[i][4]])
                y=np.append(y,Z)

            # reshaping x and y matrices
            X=X.reshape(len(X)//4,4)
            Xtrain=np.insert(X, 0, [1], axis=1)
            ytrain=y.reshape(-1,1)

            pass

        return [Xtrain,ytrain]

    # split data into training validation testing 7:1:2
    def data_split_7_1_2(self,X,y):
        Xtrain=np.array([])
        ytrain=np.array([])
        Xvalidation=np.array([])
        yvalidation=np.array([])
        Xtest=np.array([])
        ytest=np.array([])

        val=len(X)//10
        for i in range(len(X)):
            if i<7*val:
                Xtrain=np.append(Xtrain,X[i])
                ytrain=np.append(ytrain,y[i])
            elif i>=7*val and i<8*val:
                Xvalidation=np.append(Xvalidation,X[i])
                yvalidation=np.append(yvalidation,y[i])
            else:
                Xtest=np.append(Xtest,X[i])
                ytest=np.append(ytest,y[i])

        # reshaping x and y matrices for the 3 sets
        Xtrain=Xtrain.reshape(len(Xtrain)//5,5)
        Xvalidation=Xvalidation.reshape(len(Xvalidation)//5,5)
        Xtest=Xtest.reshape(len(Xtest)//5,5)
        ytrain=ytrain.reshape(-1,1)
        yvalidation=yvalidation.reshape(-1,1)
        ytest=ytest.reshape(-1,1)
        # return all objects into list
        return [Xtrain,Xvalidation,Xtest,ytrain,yvalidation,ytest]


class MyLinearRegression():

    def __init__(self,alpha=0.005,iterations=1000):
        # initialising attributes in constructor
        self.THETA=None
        self.ERROR=1000000
        self.XTRAINING_SET=None
        self.XVALIDATION_SET=None
        self.YTRAINING_SET=None
        self.YVALIDATION_SET=None
        # initialising K=5 default
        self.K=5
        self.alpha=alpha
        self.iterations=iterations
        self.J_history=np.zeros((self.iterations,1))
        self.J_history_valid=np.zeros((self.iterations,1))
        self.all_fold_thetas=[]
        self.all_fold_errors=[]
        


    def Normal_form(self,Xtrain,ytrain,K=5):
        # calling Kfold to find training and validation set for Kfold, for MAE 
        self.KFold(Xtrain,ytrain,K,1)

        a=np.matmul(Xtrain.transpose(),Xtrain)
        b=np.linalg.inv(a)
        c=np.matmul(Xtrain.transpose(),ytrain)
        self.THETA=np.matmul(b,c)
        # applying normal form to find optimal paramters

        # predicting values for both training and validations sets
        YPRED1=self.predict(self.XTRAINING_SET)
        YPRED2=self.predict(self.XVALIDATION_SET)

        # training loss vs iterations
        J_loss_training=0
        for i in range(len(YPRED1)):
            J_loss_training+=abs(YPRED1[i]-self.YTRAINING_SET[i])
        J_loss_training=J_loss_training/len(YPRED1)

        # validation loss vs iterations
        J_loss_validation=0
        for i in range(len(YPRED2)):
            J_loss_validation+=abs(YPRED2[i]-self.YVALIDATION_SET[i])
        J_loss_validation=J_loss_validation/len(YPRED2)

        print("MAE Training loss for best fold= ",J_loss_training)
        print("MAE Validation loss for best fold= ",J_loss_validation)

        


    def RMSE_vs_MAE(self,Xtrain,ytrain,K=5):
        temp=deepcopy(self.THETA)
        # Kfold for RMSE, and then keeping track of all errors for every fold
        self.KFold(Xtrain,ytrain,K,0)
        Elist1=self.all_fold_errors
        E1=sum(self.all_fold_errors)/len(self.all_fold_errors)
        m1=min(self.all_fold_errors)

        # Kfold for MAE, and then keeping track of all errors for every fold
        self.THETA=temp
        self.KFold(Xtrain,ytrain,K,1)
        Elist2=self.all_fold_errors
        E2=sum(self.all_fold_errors)/len(self.all_fold_errors)
        m2=min(self.all_fold_errors)

        # print RMSE and MAE errors
        print("RMSE for folds= ",Elist1)
        print("MAE for folds= ",Elist2)
        print("RMSE average= ",E1)
        print("MAE average= ",E2)
        print("RMSE best value= ",m1)
        print("MAE best value= ",m2)
        return self


        
    def plot_training_vs_validation(self,Xtrain,ytrain,K=5,flag=0):
        
        self.KFold(Xtrain,ytrain,K,flag)
        # error depends on flag value
        if flag==0:
            self.fit_rmse_XXX(self.XTRAINING_SET,self.YTRAINING_SET)
        else:
            self.fit_mae_XXX(self.XTRAINING_SET,self.YTRAINING_SET)
        # plot both
        training_error_vs_iterations=self.J_history
        validation_error_vs_iterations=self.J_history_valid
        
        plt.plot(training_error_vs_iterations,'b-', label="Training")
        plt.plot(validation_error_vs_iterations,'r-', label="Validation")
        plt.legend()
        plt.show()



        
        return self


    def KFold(self,Xtrain,ytrain,K=5,flag=0):
        MIN=10000000
        self.ERROR=10000000
        self.all_fold_errors=[]
        self.all_fold_thetas=[]
        LL=len(Xtrain[0])
        LY=len(ytrain[0])

        # creating fold size depending on K
        delta=len(Xtrain)//K
        batch_size1=0
        batch_size2=delta

        for i in range(K):
            XTRAIN=np.array([])
            XTEST=np.array([])
            YTRAIN=np.array([])
            YTEST=np.array([])
            
            # split data into training or validation sets depending on batch window that slides to the right
            for j in range(len(Xtrain)):
                if j<batch_size1 or j>=batch_size2-1:
                    XTRAIN=np.append(XTRAIN,Xtrain[j])
                    YTRAIN=np.append(YTRAIN,ytrain[j])
                else:
                    XTEST=np.append(XTEST,Xtrain[j])
                    YTEST=np.append(YTEST,ytrain[j])
            # resizing matrices
            XTRAIN=XTRAIN.reshape(len(XTRAIN)//LL,LL)
            XTEST=XTEST.reshape(len(XTEST)//LL,LL)
            YTRAIN=YTRAIN.reshape(len(YTRAIN)//LY,LY)
            
            # flag=0, rmse      flag=1,mae
            if flag==0:
                self.fit(XTRAIN,YTRAIN)
            else:
                self.fit_mae(XTRAIN,YTRAIN)
            # keeping track of optimised theta for every fold
            self.all_fold_thetas.append(self.THETA)
            # predict value to find errors
            YPRED=self.predict(XTEST)
            YPRED=YPRED.reshape(-1,1)

            # flag=0, rmse      flag=1,mae
            if flag==0:
                E=0
                for j in range(len(YPRED)):
                    E+=(YPRED[j]-YTEST[j])**2
                E=(E/len(YPRED))**0.5
            else:
                E=0
                for j in range(len(YPRED)):
                    E+=abs(YPRED[j]-YTEST[j])
                E=(E/len(YPRED))

            # if it's the best error observed till now, save it and the data division for this fold
            if E<self.ERROR:
                self.ERROR=E
                self.XTRAINING_SET=XTRAIN
                self.YTRAINING_SET=YTRAIN
                self.XVALIDATION_SET=XTEST
                self.YVALIDATION_SET=YTEST

            self.all_fold_errors.append(E)
            # moving the fold window to the right
            batch_size1+=delta
            batch_size2+=delta

        # take index of min error and use the paramaters (theta) for that error
        ind=self.all_fold_errors.index(self.ERROR)
        self.THETA=self.all_fold_thetas[ind]

        return self



    # cost function for RMSE
    def cost_func_rmse(self,Xtrain,ytrain):
        if len(Xtrain)==0:
            return 0
        J=0
        # predicted values
        h_x=np.matmul(Xtrain,self.THETA)
        d=h_x-ytrain
        # squares error for every row in dataset
        for i in range(len(d)):
            J+=d[i,0]**2
        # takes mean and root
        J=J/(len(ytrain))
        return J**0.5

    # cost function for RMSE
    def cost_func_mae(self,Xtrain,ytrain):
        if len(Xtrain)==0:
            return 0
        J=0
        h_x=np.matmul(Xtrain,self.THETA)
        # predicted values
        d=ytrain-h_x
        # adds absolute error values over entire dataset
        for i in range(len(d)):
            J+=abs(d[i,0])
        # takes mean
        J=J/(len(ytrain))
        return J
    

    # cost function for RMSE
    def cost_func(self,Xtrain,ytrain):
        if len(Xtrain)==0:
            return 0
        J=0
        h_x=np.matmul(Xtrain,self.THETA)
        d=h_x-ytrain
        
        # squares error for every row in dataset
        for i in range(len(d)):
            J+=d[i,0]**2
        J=J/(len(ytrain))
        return J

    # Gradiest descent algorithm for RMSE
    def gradient_descent_rmse(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        # repeats for epoch
        for i in range(self.iterations):
            right=np.matmul(Xtrain,self.THETA)-ytrain
            term=np.matmul(np.transpose(Xtrain),right)
            # "term" is the term to be subtracted from theta matrix in every epoch (derivative of cost mutltiplied by some constant)
            # here I've used a vectorized approach to calculate derivative of cost function
            term=(term*self.alpha)/(m*  (self.cost_func_rmse(Xtrain,ytrain))  )

            # subtracting the term from theta
            self.THETA=self.THETA-term
            # saving cost vs iteration value in AA
            AA=np.append(AA,self.cost_func_rmse(Xtrain,ytrain))

        # saving J history for every epoch in class attribute
        self.J_history=AA
        self.J_history=self.J_history.reshape(-1,1  )
        return self

    def gradient_descent_rmse_XXX(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        BB=np.array([])
        # repeats for epoch
        for i in range(self.iterations):
            right=np.matmul(Xtrain,self.THETA)-ytrain
            term=np.matmul(np.transpose(Xtrain),right)
            # "term" is the term to be subtracted from theta matrix in every epoch (derivative of cost mutltiplied by some constant)
            # here I've used a vectorized approach to calculate derivative of cost function
            term=(term*self.alpha)/(m*  (self.cost_func_rmse(Xtrain,ytrain))  )

            # subtracting the term from theta
            self.THETA=self.THETA-term
            # saving cost vs iteration value in AA
            AA=np.append(AA,self.cost_func_rmse(Xtrain,ytrain))
            BB=np.append(BB,self.cost_func_rmse(self.XVALIDATION_SET,self.YVALIDATION_SET  ))

        # saving J history for every epoch in class attribute
        self.J_history=AA
        self.J_history_valid=BB
        self.J_history=self.J_history.reshape(-1,1  )
        self.J_history_valid=self.J_history_valid.reshape(-1,1  )
        return self







    def gradient_descent_mae(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        
        # repeats for epoch
        for i in range(self.iterations):
            right=ytrain-np.matmul(Xtrain,self.THETA)
            bbb=deepcopy(Xtrain)
            # for the derivative of modulus function, we need to see what the sign of the entity is, here I am using a vectorized approach to find derivate term
            # and for that I need to add values in X matrix down all the columns and adjust the sums using the sign obtained from entity value
            # if value<=0 , I have multiplied the whole copy of row with -1 so when I add this row I essentially subtract values
            for i in range(len(Xtrain)):
                if right[i]>0:
                    bbb[i]=bbb[i]*-1
            # columnwise sum and resizing matrix
            Xi=bbb.sum(axis=0)
            Xi=Xi.reshape(len(Xi),1)
            # subtracting derivative term from theta
            self.THETA=self.THETA-((self.alpha*Xi)/len(ytrain))
            # saving cost vs iteration value in AA
            AA=np.append(AA,self.cost_func_mae(Xtrain,ytrain))

    def gradient_descent_mae_XXX(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        BB=np.array([])
        
        # repeats for epoch
        for i in range(self.iterations):
            right=ytrain-np.matmul(Xtrain,self.THETA)
            bbb=deepcopy(Xtrain)
            # for the derivative of modulus function, we need to see what the sign of the entity is, here I am using a vectorized approach to find derivate term
            # and for that I need to add values in X matrix down all the columns and adjust the sums using the sign obtained from entity value
            # if value<=0 , I have multiplied the whole copy of row with -1 so when I add this row I essentially subtract values
            for i in range(len(Xtrain)):
                if right[i]>0:
                    bbb[i]=bbb[i]*-1
            # columnwise sum and resizing matrix
            Xi=bbb.sum(axis=0)
            Xi=Xi.reshape(len(Xi),1)
            # subtracting derivative term from theta
            self.THETA=self.THETA-((self.alpha*Xi)/len(ytrain))
            # saving cost vs iteration value in AA
            AA=np.append(AA,self.cost_func_mae(Xtrain,ytrain))
            BB=np.append(BB,self.cost_func_mae(self.XVALIDATION_SET,self.YVALIDATION_SET))
    

        # saving J history for every epoch in class attribute
        self.J_history=AA
        self.J_history_valid=BB
        self.J_history=self.J_history.reshape(-1,1  )
        self.J_history_valid=self.J_history_valid.reshape(-1,1  )
        return self


    def gradient_descent(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        # repeats for epoch
        for i in range(self.iterations):
            right=np.matmul(Xtrain,self.THETA)-ytrain
            # "term" is the term to be subtracted from theta matrix in every epoch (derivative of cost mutltiplied by some constant)
            # here I've used a vectorized approach to calculate derivative of cost function
            term=np.matmul(np.transpose(Xtrain),right)
            term=(term*2*self.alpha)/m
            # subtracting the term from theta
            self.THETA=self.THETA-term
            # saving cost vs iteration value in AA
            AA=np.append(AA,self.cost_func(Xtrain,ytrain))

        # saving J history for every epoch in class attribute
        self.J_history=AA
        self.J_history=self.J_history.reshape(-1,1  )
        return self
    

 

    # fit my model for RMSE, takes in Xtrain set and returns otpimised Theta for the model by training it
    def fit(self, X, y):
        # initialising theta matrix with 0s and same size of Xtrain
        self.THETA=np.ones((len(X[0]),1))
        # applying gradient descent algorithm to find our optimised Theta for RMSE
        self.gradient_descent_rmse(X,y)
        return self

    # fit my model for MAE, takes in Xtrain set and returns otpimised Theta for the model by training it
    def fit_mae(self, X, y):
        # initialising theta matrix with 0s and same size of Xtrain
        self.THETA=np.ones((len(X[0]),1))
        # applying gradient descent algorithm to find our optimised Theta for MAE
        self.gradient_descent_mae(X,y)
        return self
    def fit_mae_XXX(self, X, y):
        # initialising theta matrix with 0s and same size of Xtrain
        self.THETA=np.ones((len(X[0]),1))
        # applying gradient descent algorithm to find our optimised Theta for MAE
        self.gradient_descent_mae_XXX(X,y)
        return self
    def fit_rmse_XXX(self, X, y):
        # initialising theta matrix with 0s and same size of Xtrain
        self.THETA=np.ones((len(X[0]),1))
        # applying gradient descent algorithm to find our optimised Theta for MAE
        self.gradient_descent_rmse_XXX(X,y)
        return self

    # predict my model for a test set, using the thetas I have optimised after training my model for X
    def predict(self, X):
        # predicting values by multiplying X matrix by our optimised Theta and then resizing
        ypred=(np.matmul(X,self.THETA))
        ypred=ypred.reshape(1,len(ypred))
        return ypred[0]


class MyLogisticRegression():

    def __init__(self,alpha=0.01,iterations=500, X_V_SET=None,Y_V_SET=None):
        # class attributes defined in the constructor with 2 default values
        self.THETA=None
        self.alpha=alpha
        self.iterations=iterations
        self.J_history=np.zeros((self.iterations,1))
        self.J_history_valid=np.zeros((self.iterations,1))
        self.X_V_SET=X_V_SET
        self.Y_V_SET=Y_V_SET
        
        pass

    # used to find apply sigmoid function on z, could be a matrix so we iterate over every element and apply operation
    def sigmoid(self,z):
        for i in range(len(z)):
            for j in range(len(z[0])):
                # used try except block to handle overflow scenarios
                try:
                    z[i,j]=1/(1+np.exp(-z[i,j]))
                except ValueError:
                    if -z[i,j]>600:
                        z[i,j]=0
                    elif -z[i,j]<-600:
                        z[i,j]=1
                
        return z
    
    # calculates log loss for my model by iterating over every row and filling in values
    def cost_func_logistic(self,Xtrain,ytrain):
        if len(Xtrain)==0:
            return 0
        J=0
        for i in range(len(Xtrain)):
            temp=np.matmul(-Xtrain[i],self.THETA)
            # used try except block to handle overflow scenarios
            try:
                z=1/(1+np.exp(temp))
            except ValueError:
                if temp>600:
                    z=0
                elif temp<-600:
                    z=1
            # impose heavy penalty when overflow happens
            if ytrain[i]==0:
                try:
                    J+=-log(1-z)
                except ValueError:
                    J+=40
            else:
                try:
                    J+=-log(z)
                except ValueError:
                    J+=40

        J=J/len(Xtrain)
        return J



    # batch gradient descent algorithm
    # takes the entire dataset while calculating derivative term of cost function and then reduces it from the Theta in one epoch
    def batch_gradient_descent(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        BB=np.array([])
        # epoch
        for i in range(self.iterations):

            A=self.sigmoid(np.matmul(Xtrain,self.THETA))
            right=A-ytrain
            term=np.matmul(np.transpose(Xtrain),right)
            # subtracting derivative term from theta
            term=(term*2*self.alpha)/m
            self.THETA=self.THETA-term
            AA=np.append(AA,self.cost_func_logistic(Xtrain,ytrain))
            BB=np.append(BB,self.cost_func_logistic(self.X_V_SET,self.Y_V_SET))

        # saving J history into class attribute
        self.J_history=AA
        self.J_history_valid=BB
        self.J_history=self.J_history.reshape(-1,1  )
        self.J_history_valid=self.J_history_valid.reshape(-1,1  )
        return self        

    # stochastic gradient descent algorithm
    # takes one random row from entire dataset while calculating derivative term of cost function and then reduces it from the Theta in one epoch
    def stochastic_gradient_descent(self,Xtrain,ytrain):
        m=len(ytrain)
        AA=np.array([])
        BB=np.array([])
        DAD_X=deepcopy(Xtrain)
        # shuffle copy of Xtrain to give a random row for loop
        np.random.seed(14) 
        np.random.shuffle(DAD_X)
        
        # shuffle copy of ytrain to give a random row for loop
        DAD_Y=deepcopy(ytrain)
        np.random.seed(14) 
        np.random.shuffle(DAD_Y)


        for i in range(self.iterations):
            # using % to roll over the list
            temp=np.matmul(DAD_X[i%len(DAD_X)],self.THETA)
            temp=temp.reshape(-1,1)

            A=self.sigmoid( temp)
            right=A-DAD_Y[i%len(DAD_Y)]

            CC=deepcopy(DAD_X[i%len(DAD_X)])
            CC=CC.reshape(len(CC),1)
            term=np.matmul(CC,right)
            # subtracting derivative term from theta

            term=(term*2*self.alpha)
            self.THETA=self.THETA-term
            AA=np.append(AA,self.cost_func_logistic(DAD_X,DAD_Y))
            BB=np.append(BB,self.cost_func_logistic(self.X_V_SET,self.Y_V_SET))

        # saving J history into class attribute
        self.J_history=AA
        self.J_history_valid=BB
        self.J_history=self.J_history.reshape(-1,1  )
        self.J_history_valid=self.J_history_valid.reshape(-1,1  )
        return self        



    # fit my model for given X and Y to find optimised parameters for Batch GD
    def fit(self, X, y):
        # initialise theta with zeros, of the size of x
        self.THETA=np.ones((len(X[0]),1))
        # apply batch gradient descent on X to optimise given thetas
        self.batch_gradient_descent(X,y)
        return self

    # fit my model for given X and Y to find optimised parameters for Stochastic GD
    def fit_stochastic(self, X, y):
        # initialise theta with zeros, of the size of x
        self.THETA=np.ones((len(X[0]),1))
        # apply stochastic gradient descent on X to optimise given thetas
        self.stochastic_gradient_descent(X,y)
        return self

    # fit my model on testing data using our optimised parameters
    def predict(self, X):
        yy=np.zeros((len(X),1))

        A=self.sigmoid(np.matmul(X,self.THETA))
        # calculating or h_theta

        # we iterate over all values of X, and for biclassification (class 0 and 1), we check if value>=0.5 we classify it as 1, else 0
        for i in range(len(X)):
            if A[i]>=0.5:
                yy[i]=1
            else:
                yy[i]=0
        return yy

    # function to calculate Accuracy for our testing set using our predicted values
    def cal_accuracy(self,y1,y2):
        c=0
        # we add to count when the class label that we classify using our algorithm is equal to the actual label
        for i in range(len(y1)):
            if y1[i]==y2[i]:
                c+=1
        print("Accuracy= ",(c*100)/len(y2))
        return None
    
    # function to plot training error vs iterations, and validation error vs iterations
    def plot_training_vs_validations_iterations(self,training_error_vs_iterations,validation_error_vs_iterations):
        temp=np.array([])
        for i in range(self.iterations):
            temp=np.append(temp,i+1)

        # plt.subplot(1,2,1)
        # plt.title('Training loss')
        
        # plt.subplot(1,2,2)
        plt.plot(training_error_vs_iterations,'b-', label="Training")
        plt.plot(validation_error_vs_iterations,'r-', label="Validation")
        plt.legend()
        plt.show()
        # plt.title('Validation loss')

        return self