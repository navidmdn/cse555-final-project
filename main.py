#!/usr/bin/env python
# coding: utf-8

# # Initializations

# ## imports

# In[ ]:


from torchvision.datasets import MNIST
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

import torch
import torchvision

import numpy as np


# ## configs

# In[ ]:


batch_size_train = 128
batch_size_test = 100
random_seed = 12453211


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


# ## Creating noisy datasets

# # Models

# ## SVM

# In[ ]:





# ## Logistic Regression

# ### preprocessing data

# In[ ]:


#TODO: flattening the entries    


# ### defining model

# In[ ]:


svm = LogisticRegression(
    penalty='l2',
    n_jobs=8,
    max_iter=100,
    multi_class='ovr',
    random_state=random_seed
)


# ### training

# In[ ]:


#TODO: search on a few hyperparameters and tune on validation set


# ### results

# In[ ]:





# # Validation

# In[ ]:




