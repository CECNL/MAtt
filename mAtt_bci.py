import torch
import torch.nn as nn
from utils.functions import trainNetwork, testNetwork
from mAtt.mAtt import mAtt_bci
from utils.GetBci2a import getAllDataloader
import os
import argparse


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=int, default=1, help='subjectxx you want to triain')
    ap.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    ap.add_argument('--iterations', type=int, default=350, help='number of training iterations')
    ap.add_argument('--epochs', type=int, default=3, help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--bs', type=int, default=128, help='batch size')
    ap.add_argument('--model_path', type=str, default='./checkpoint/bci2a/', help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='data/BCICIV_2a_mat/', help='data path')
    args = vars(ap.parse_args())

    print(f'subject{args["sub"]}')
    trainloader, validloader, testloader = getAllDataloader(subject=args['sub'], 
                                                            ratio=8, 
                                                            data_path=args['data_path'], 
                                                            bs=args['bs'])
    net = mAtt_bci(args['epochs']).cpu()

    args.pop('bs')
    args.pop('data_path')
    trainNetwork(net, 
                trainloader, 
                validloader, 
                testloader,
                **args
                )
    net = torch.load(os.path.join(args["model_path"], f'repeat{args["repeat"]}_sub{args["sub"]}_epochs{args["epochs"]}_lr{args["lr"]}_wd{args["wd"]}.pt'))    
    acc = testNetwork(net, testloader)    
    print(f'{acc*100:.2f}')

