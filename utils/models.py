import torch
from torch import nn
from selfonn import SelfONNLayer
#from selfonn import SelfONNLayer
import matplotlib.pyplot as plt

def get_model(args):
    if args.model=='dncnn': return DnCNN(num_of_layers=args.num_layers)
    elif args.model=='selfdncnn': return SelfDnCNN(num_of_layers=args.num_layers,q=args.q)
        


class SelfDnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17,q=3):
        super(SelfDnCNN, self).__init__()
        #print("SelfDNCNN initialized with q",q,num_of_layers)
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(SelfONNLayer(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False,q=q))
        layers.append(nn.Tanh())
        for _ in range(num_of_layers-2):
            layers.append(SelfONNLayer(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False,q=q))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.Tanh())
        layers.append(SelfONNLayer(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False,q=q))
        layers.append(nn.Tanh())
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out


