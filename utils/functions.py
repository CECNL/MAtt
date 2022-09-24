import torch
import torch.nn as nn
import sys
import os
sys.path.append("..") 
from mAtt.optimizer import MixOptimizer
from sklearn.metrics import roc_auc_score as ras
import numpy as np

def trainNetwork(net, trainloader, validloader, testloader, model_path=None, iterations=500, lr=5*1e-4, wd=None, repeat=None, sub=None, epochs=None):
    softmax = nn.Softmax(dim=1)
    CE = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
    optimizer = MixOptimizer(optimizer)
    bestLoss = 1e10
    val_len = len(validloader)
    
    for ite in range(iterations):
        net.train()
        acc_val = 0
        acc_tr = 0
        tr_len = 0
        for xb, yb in trainloader:
            tr_len += yb.shape[0]
            out = net(xb)
            loss = CE(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_tr += (torch.max(out, 1).indices==yb).sum().item()
        
        net.eval()
        TL = 0
        for xb, yb in validloader:
            with torch.no_grad():
                out = net(xb)
                if torch.argmax(softmax(out)) == yb:
                    acc_val += 1

                TL += (CE(out, yb).item())
        print('')
        print(f'Iteration{ite}=====')
        print(f'train_loss:{loss:.4f}    val_loss:{TL/val_len:.4f}')
        print(f'train_acc:{acc_tr/tr_len:.4f}    val_acc:{acc_val/val_len:.4f}')
                
        if TL < bestLoss:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print(f'create {model_path}.......')
            bestLoss = TL
            final_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt')
            print(f'saving to {final_path}')
            torch.save(net, final_path)
            testnet = torch.load(final_path)
            test_acc = testNetwork(testnet, testloader)
            print(f'test_acc:{test_acc}')
    net = torch.load(os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt'))
    return net  

def testNetwork(net, testloader):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    for xb, yb in testloader:
        with torch.no_grad():
            pred = net(xb)
            if torch.argmax(softmax(pred)) == yb:
                acc += 1
    
    return acc / len(testloader)

def testNetwork_auc(net, testloader):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    y_pred = torch.empty(0)
    y_true = torch.empty(0)
    for xb, yb in testloader:
        with torch.no_grad():
            pred = net(xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)
            
    return ras(y_true.detach().numpy(), y_pred.detach().numpy())


