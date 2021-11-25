# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Initializations
# %% [markdown]
# ## imports

# %%
from torchvision.datasets import MNIST
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import torch
import torchvision

import seaborn as sns
import numpy as np

# %% [markdown]
# ## configs

# %%
batch_size_train = 128
batch_size_test = 100
random_seed = 12453211

imbalanced_weights = {
    0: 0.3,
    1: 0.3,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.3,
    6: 1.0,
    7: 0.3,
    8: 1.0,
    9: 1.0
}

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# %% [markdown]
# # Data Preparation
# %% [markdown]
# ## Loading torch dataset

# %%

train_set = torchvision.datasets.MNIST(
    '.',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

test_set = torchvision.datasets.MNIST(
    '.',
    train=False, 
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
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
    test_set,
    batch_size=batch_size_test,
    shuffle=False
)

# %% [markdown]
# ## loading numpy dataset

# %%
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

# %% [markdown]
# ## distribution of classes

# %%
sns.histplot(train_y, bins=[i for i in range(11)])
sns.histplot(test_y, bins=[i for i in range(11)])

# %% [markdown]
# ## Creating imbalanced dataset

# %%
def make_imbalanced(ds_x, ds_y, imbalanced_weights=imbalanced_weights):
    class_partition = {k:[] for k in range(10)}

    for x, y in zip(ds_x, ds_y):
        class_partition[y].append((x, y))

    for i in range(10):
        idxs = np.random.randint(0, len(class_partition[i]), int(imbalanced_weights[i]*len(class_partition[i])))
        class_partition[i] = [class_partition[i][j] for j in idxs]
        print(f"class {i}: size={len(class_partition[i])}")

    imbalanced_train = []

    for partition in class_partition.values():
        imbalanced_train.extend(partition)

    np.random.shuffle(imbalanced_train)
    imbalanced_train_x, imbalanced_train_y = zip(*imbalanced_train)
    
    return imbalanced_train_x, imbalanced_train_y

imb_train_x, imb_train_y = make_imbalanced(train_x, train_y)


# %%
sns.histplot(imb_train_y, bins=[i for i in range(11)])

# %% [markdown]
# ## create a dataset with symmetric noise

# %%
def apply_sym_noise(ds_x):
    pass

# %% [markdown]
# ## create a dataset with asymmetric noise

# %%
def apply_asym_noise(ds_x):
    pass

# %% [markdown]
# # Models
# %% [markdown]
# ## validation functions
# %% [markdown]
# ### draw confusion matrix

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def conf_mat(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True)


# %%
from sklearn.metrics import classification_report

def clf_metrics(y_true, y_pred, n_class=10):
    class_names = [str(i) for i in range(n_class)]
    print(classification_report(y_true, y_pred))
    

# %% [markdown]
# ## SVM
# %% [markdown]
# ### preprocessing data

# %%
def preprocess(x, y):
    x, y = x.squeeze(), y
    return x.reshape((x.shape[0], -1)), y

train_x, train_y = preprocess(train_x, train_y)
test_x, test_y = preprocess(test_x, test_y)
valid_x, valid_y = preprocess(valid_x, valid_y)

train_x.shape

# %% [markdown]
# ### model definition

# %%
svm = SVC(
    kernel='linear',
    decision_function_shape='ovr',
    random_state=random_seed,
    verbose=True,
) 

svm.fit(train_x, train_y)
y_pred = svm.predict(test_x)


# %%
svm.coef_.shape


# %%
conf_mat(test_y, y_pred)

# %% [markdown]
# ### model report

# %%
clf_metrics(test_y, y_pred)

# %% [markdown]
# ## LDAM

# %%



