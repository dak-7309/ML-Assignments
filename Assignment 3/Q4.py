# -*- coding: utf-8 -*-
"""MLq4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iwB-efabWFcygY0BBBHqmpVtEvbb-CiE
"""

# from google.colab import drive
# drive.mount('/content/drive/')

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn import decomposition
from sklearn import manifold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import time
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
# importing all pytorch and necessary libraries

# alexnet model pretrained on ImageNet
alexnet = models.alexnet(pretrained=True)
SEED = 1234

# seed values for consistent results
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# obtaining training and testing data, 3072 dimensional
train_data=pd.read_pickle('train_CIFAR.pickle')
test_data=pd.read_pickle('test_CIFAR.pickle')

# this methods reshapes images to obtain 32x32x3 from 3072, i.e. for 3 channels RGB
def reshape_image(image):
    assert image.shape[1]==3072
    dimension=np.sqrt(1024).astype(int)
    Red=image[:,0:1024].reshape(image.shape[0],dimension,dimension,1)
    Green=image[:,1024:2048].reshape(image.shape[0],dimension,dimension,1)
    Blue=image[:,2048:3072].reshape(image.shape[0],dimension,dimension,1)
    photo=np.concatenate([Red,Green,Blue],-1)
    return photo

# reshaping my dataset 
X_train=reshape_image(train_data["X"])
X_test=reshape_image(test_data["X"])
y_test=np.array(test_data["Y"])
y_train=np.array(train_data["Y"])

# applying EDA to obtain clss distribution
sns.countplot(y_train)
histogram_=pd.Series(y_train).groupby(y_train).count()
print(histogram_)
# yields histogram that shows class distribution, here 5000 5000 each for the 2 classes

# further used SVD and scatterplot to visualize my dataset to distinguish between the 2 classes as can be seen in the figure
svd=TruncatedSVD(n_components=2,random_state=32)
X_SVD=svd.fit_transform(train_data["X"])
sns.scatterplot(X_SVD[:,0],X_SVD[:,1],hue=train_data["Y"],legend='full',palette=sns.color_palette("hls",2))
plt.show()

# defined a custom class that has the desired methods __init,__getitem__ abd __len__ needed for the DataLoader Class
class MyClass(Dataset): 
    def __init__(self,data,label,transform=None):
        self.data=data
        self.label=label
        self.img_shape=data.shape
        self.transform=transform
        # attributes are data, labels, transform for the data(image here) and the dimensions of data
        
    def __getitem__(self,ind):
        image=Image.fromarray(self.data[ind])
        label=self.label[ind]

        if self.transform is not None:
            image=self.transform(image)
        else:
            image_tensor=transforms.ToTensor()
            image=image_tensor(image)
        return image,label
        # returns data and labels for given index or rowm in the tensor format
        
    def __len__(self):
        # returns length of data
        return len(self.data)

# Preprocessing steps

# method that normalises my data according to the mean and standard deviation values
def func_normalise(data):
    Mean=data.mean(axis=(0,1,2))/255.0
    Std_dev=data.std(axis=(0,1,2))/255.0
    output=transforms.Normalize(mean=Mean,std=Std_dev)
    return output

# applies resize transform and tensor type conversion for dataset, also calls the normalize method
train_transform_aug=transforms.Compose([transforms.Resize(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),func_normalise(X_train)])
test_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),func_normalise(X_test)])

# creates training and validation objects of MyClass needed for DataLoader
trainset=MyClass(data=X_train,label=y_train,transform=train_transform_aug)
testset=MyClass(data=X_test,label=y_test,transform=test_transform)

BATCH_SIZE=64
NUM_WORKERS=1

train_loader=DataLoader(dataset=trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
test_loader=DataLoader(dataset=testset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

# my custom neural network class that contains information about my network structure
# and has a forward pass method that applies activation of my data
class MyNeuralNetwork(nn.Module):
  def __init__(self, inp_size=1000, out_size=2, hidden_size_1=512, hidden_size_2=256):
      # here we have 2 hidden layers to extra arguments
      super().__init__()
      
      # network structure
      self.input_fc=nn.Linear(inp_size,hidden_size_1)
      self.hidden_fc=nn.Linear(hidden_size_1,hidden_size_2)
      self.output_fc=nn.Linear(hidden_size_2,out_size)
        
  def forward(self, X):
      batch_size=X.shape[0]
      X=X.view(batch_size,-1)
      # 2 hidden layers and hence 2 activations called on data
      h_1=F.relu(self.input_fc(X))
      h_2=F.relu(self.hidden_fc(h_1))
      y_pred=self.output_fc(h_2)
      # returns data post activation and labels
      return y_pred,h_2

# train is used to train my model for the provided iterator and performs forward and backward
# prop and takes into consideration the inputted optimizer, criterion and device
def train(model, iterator, optimizer, criterion, device):

    epoch_loss=0
    epoch_acc=0
    model.train()
    
    for (x,y) in iterator:
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        alexnet.to(device)
        # will apply my neural network model on alexnet model, that is applied on x
        y_pred,_=model(alexnet(x).float())
        loss=criterion(y_pred,y)
        acc=calculate_accuracy(y_pred, y)
        # computes accuracy given true and predicted labels
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        epoch_acc+=acc.item()

    # returns loss and accuracy per batch
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

# evaluate is used for testing data for the provided iterator and performs forward 
# prop and takes into consideration the inputted optimizer, criterion and device
def evaluate(model, iterator, criterion, device):
    
    epoch_loss=0
    epoch_acc = 0
    model.eval()
    y_predictions=[]
    y_real=[]
    probabilities=[]
    # lists that will contain, true y labels, predicted y labels, and predicted y probabilities
    
    with torch.no_grad():
        for (x,y) in iterator:
            x=x.to(device)
            y=y.to(device)
            alexnet.to(device)
            # will apply my neural network model on alexnet model, that is applied on x
            y_pred,_=model(alexnet(x).float())
            loss=criterion(y_pred,y)
            acc=calculate_accuracy(y_pred, y)
            # computes accuracy given true and predicted labels

            HIGH_PRED=y_pred.argmax(1,keepdim=True)
            y_predictions.append(HIGH_PRED)
            # contains predicted y labels
            y_real.append(y)
            # contains true y labels
            y_prob=F.softmax(y_pred,dim=-1)
            # applied softmax classifier to get probabilities
            probabilities.append(y_prob)
            # contains predicted y probabilities

            epoch_loss+=loss.item()
            epoch_acc+=acc.item()
    # combines the labels and probabilities over iterators and puts them in a collection
    y_predictions=torch.cat(y_predictions,dim=0)
    y_real=torch.cat(y_real,dim=0)
    probabilities=torch.cat(probabilities,dim=0)
    # return loss and accuracy per batch, also the collections for true labels, predicted labels and predicted label probabilities
    return epoch_loss/len(iterator),epoch_acc/len(iterator),y_predictions.cpu(),y_real.cpu(),probabilities.cpu()

def calculate_accuracy(y_pred,y):
    # returns accuracy given corresponding y and y predicted
    row=y.shape[0]
    HIGH_PRED=y_pred.argmax(1,keepdim=True)
    correct=HIGH_PRED.eq(y.view_as(HIGH_PRED))
    val=correct.sum()
    Accuracy=val.float()/row
    return Accuracy

# initialies my model and runs it on gpu if its available as the code execution takes time
model=MyNeuralNetwork()
if torch.cuda.is_available():
    model.cuda()
    alexnet.cuda()

# cross entropy loss is the criterion as asked
criterion=nn.CrossEntropyLoss()
# adam model's parameters with custom learning rate as provided in argument
optimizer=optim.Adam(model.parameters())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
criterion=criterion.to(device)

EPOCHS=10

# train model trains data on the neural network model
def Train_model(EPOCHS,model,train_loader, optimizer, criterion, device):
  for epoch in range(EPOCHS): 
      _,train_accuracy=train(model,train_loader,optimizer,criterion,device)
      print(epoch, 100*train_accuracy)
      # prints accuracy per epoch

Train_model(EPOCHS,model,train_loader, optimizer, criterion, device)

_,test_accuracy,y_predictions,y_real,probabilities=evaluate(model,test_loader,criterion,device)
# evaluate method returns collections for true labels, predicted labels and predicted label probabilities

# this method converts type to numpy so I can applly sklearn models on it for Test accuracy, ROC Curve and Confusion matrix
def to_numpy(test_accuracy,y_predictions,y_real,probabilities):
  y_predictions=y_predictions.numpy()
  y_real=y_real.numpy()
  probabilities=probabilities.numpy()
  probability_majo_class=probabilities[:,1]
  # we just need the probabilities corresponding to the positive class, so 2nd column
  return y_predictions,y_real,probabilities,probability_majo_class

y_predictions,y_real,probabilities,probability_majo_class=to_numpy(test_accuracy,y_predictions,y_real,probabilities)

# plots roc curve given the true y labels, predicted labels and predicted label probabilities 
def plot_ROC(y_real,y_predictions, probability_majo_class):
  # tpr=true positive rate, fpr=false positive rate, returned by the metrics as shown below
  fpr,tpr,thresholds=metrics.roc_curve(y_real,probability_majo_class)
  # applying auc on fpr and tpr using sklearns auc metric
  roc_auc=metrics.auc(fpr,tpr)
  # plots roc curve
  plt.figure()
  plt.plot(fpr,tpr,color='red',lw=2,label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1],[0, 1],color='black',lw=2, linestyle='--')
  plt.xlim([0.0,1.0])
  plt.ylim([0.0,1.05])
  plt.ylabel('TPR')
  plt.xlabel('FPR')
  plt.title('ROC curve')
  plt.legend()
  plt.show()

# creates confusion matrix given the true y labels, predicted labels
def confusion_matrix_plot(y_real,y_predictions):
  # sklearn implementation that returns matrix of size (no. of classes)x(no. of classes)
  CM=(confusion_matrix(y_real, y_predictions))
  print(CM)
  # sklearn visualization for confusion matrix for class labels 0 and 1
  GG=ConfusionMatrixDisplay(CM, np.array([0,1]))
  GG.plot()

# returns testing accuracy using sklearn's accuracy metric given true and predicted labels
def testing_accuracy(y_real,y_predictions):
  print("Testing accuracy: ",100*metrics.accuracy_score(y_real, y_predictions))

# calling the above functions for the results
plot_ROC(y_real,y_predictions, probability_majo_class)
confusion_matrix_plot(y_real,y_predictions)
testing_accuracy(y_real,y_predictions)