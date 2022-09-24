import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import io
import os

   
# session 123 is training set, session4 is validation set, and session5 is testing set. 
def getAllDataloader(subject, data_path='./data/BCIcha/', bs=64):
    dev = torch.device("cpu")
    # dev = torch.device("cuda")
    train = io.loadmat(os.path.join(data_path, f'Data_S{subject:02d}_Sess' + '.mat'))

    tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
    templabel= torch.Tensor(train['y_test']).view(-1)
    x_train=tempdata[:180]
    y_train=templabel[:180]
    
    x_valid=tempdata[180:240]
    y_valid=templabel[180:240]
    
    x_test =tempdata[240:340]
    y_test =templabel[240:340]

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)
    

    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    test_dataset = Data.TensorDataset(x_test, y_test)
    
    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size =bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )
    testloader =  Data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader, testloader
    