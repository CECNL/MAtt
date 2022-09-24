import torch
import torch.utils.data as Data
from scipy import io
import numpy as np
import os

def split_train_valid_set(x_train, y_train, ratio):
    s = y_train.argsort()
    x_train = x_train[s]
    y_train = y_train[s]

    cL = int(len(x_train) / 4)

    class1_x = x_train[ 0 * cL : 1 * cL ]
    class2_x = x_train[ 1 * cL : 2 * cL ]
    class3_x = x_train[ 2 * cL : 3 * cL ]
    class4_x = x_train[ 3 * cL : 4 * cL ]

    class1_y = y_train[ 0 * cL : 1 * cL ]
    class2_y = y_train[ 1 * cL : 2 * cL ]
    class3_y = y_train[ 2 * cL : 3 * cL ]
    class4_y = y_train[ 3 * cL : 4 * cL ]

    vL = int(len(class1_x) / ratio)

    x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
    y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

    x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
    y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

    return x_train, y_train, x_valid, y_valid



# split dataset
def getAllDataloader(subject, ratio, data_path, bs):
    train = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_T.mat'))
    test = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_E.mat'))

    x_train = torch.Tensor(train['x_train']).unsqueeze(1)
    y_train = torch.Tensor(train['y_train']).view(-1)
    x_test = torch.Tensor(test['x_test']).unsqueeze(1)
    y_test = torch.Tensor(test['y_test']).view(-1)

    x_train, y_train, x_valid, y_valid = split_train_valid_set(x_train, y_train, ratio=ratio)
    dev = torch.device('cpu')

    x_train = x_train[:, :, :, 124:562].to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid[:, :, :, 124:562].to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test[:, :, :, 124:562].to(dev)
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
        batch_size = bs,
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
        
