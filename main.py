#!/usr/bin/env python
# coding: utf-8

# # Initializations

# ## imports

# In[ ]:


from torchvision.datasets import MNIST
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import torch
import torchvision
from torch.autograd import Variable

import seaborn as sns
import numpy as np


# ## configs

# In[ ]:


batch_size_train = 128
batch_size_test = 100
random_seed = 12453211
learning_rate =  0.01

torch.manual_seed(random_seed)
np.random.seed(random_seed)


# # Data Preparation

# ## Loading torch dataset

# In[ ]:



train_set = torchvision.datasets.MNIST(
    '.',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size_train,
    shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', train=False, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
    ])),
    batch_size=batch_size_test,
    shuffle=False
)


# ## loading numpy dataset

# In[ ]:


def data_loader_to_numpy(data_loader):
    result_x = []
    result_y = []
    for x, y in data_loader:
        result_x.append(x.numpy())
        result_y.append(y.numpy())
        
    return np.concatenate(result_x, axis=0), np.concatenate(result_y, axis=0)
    
train_x, train_y = data_loader_to_numpy(train_loader)
test_x, test_y = data_loader_to_numpy(test_loader)
valid_x, valid_y = data_loader_to_numpy(valid_loader)

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


# ## distribution of classes

# In[ ]:


sns.histplot(train_y, bins=[i for i in range(11)])
sns.histplot(test_y, bins=[i for i in range(11)])


# ## Creating noisy datasets

# # Models

# ## validation functions

# ### draw confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def conf_mat(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True)


# In[ ]:


from sklearn.metrics import classification_report

def clf_metrics(y_true, y_pred, n_class=10):
    class_names = [str(i) for i in range(n_class)]
    print(classification_report(y_true, y_pred))
    


# ## SVM

# ### preprocessing data

# In[ ]:


def preprocess(x, y):
    x, y = x.squeeze(), y
    return x.reshape((x.shape[0], -1)), y

train_x, train_y = preprocess(train_x, train_y)
test_x, test_y = preprocess(test_x, test_y)
valid_x, valid_y = preprocess(valid_x, valid_y)

train_x.shape


# ### model definition

# In[ ]:


svm = SVC(
    kernel='linear',
    decision_function_shape='ovr',
    random_state=random_seed,
    verbose=True,
) 

svm.fit(train_x, train_y)
y_pred = svm.predict(test_x)


# In[ ]:


svm.coef_.shape


# In[ ]:


conf_mat(test_y, y_pred)


# ### model report

# In[ ]:


clf_metrics(test_y, y_pred)


# ## Logistic Regression

# In[ ]:


# Logistic regression model
input_size = train_x[0].shape[0]
num_classes = 10
num_epochs = 1

model = torch.nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


# In[ ]:


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# In[ ]:





# In[ ]:


# Model definition
class LogisticRegression(torch.nn.Module):
    def __init__(self,n_input_features,output_features):
        super(LogisticRegression,self).__init__()
        self.linear = torch.nn.Linear(n_input_features,output_features)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        print(y_predicted)
        return y_predicted

input_features = train_x[0].shape[0]
output_features = 10

model = LogisticRegression(input_features,output_features)


# In[ ]:


#Loss and optimizer definition
# Using binary cross entropy loss
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# Training loop
n_epochs = 1
for epoch in range(n_epochs):
    for i,(input,labels) in enumerate(train_loader):
        print(input)
        print(labels)


# ### preprocessing data

# In[ ]:


#TODO: flattening the entries    


# ### defining model

# In[ ]:





# ### training

# In[ ]:


#TODO: search on a few hyperparameters and tune on validation set


# ### results

# In[ ]:





# # Validation

# In[ ]:




