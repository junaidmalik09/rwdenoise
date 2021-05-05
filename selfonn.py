import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt


class SelfONNLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,q=1,sampling_factor=1,idx=-1,dir=[],debug=False,output=False,vis=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.q = q
        self.sampling_factor = sampling_factor
        self.weights = nn.Parameter(torch.Tensor(q,out_channels,in_channels,kernel_size,kernel_size)) # Q x C x K x D
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.q = q
        self.dir = dir
        self.debug = debug
        self.idx = idx #deprecated
        self.output = output #deprecated
        self.reset_parameters()
        
                
    def reset_parameters(self):
        #print('xavier')
        bound = 0.01
        if self.bias is not None: nn.init.uniform_(self.bias,-bound,bound)
        for q in range(self.q): nn.init.xavier_uniform_(self.weights[q])

    def reset_parameters_q(self):
        #print('xavier-q')
        bound = 0.01
        if self.bias is not None: nn.init.uniform_(self.bias,-bound,bound)
        for q in range(self.q): 
            nn.init.xavier_uniform_(self.weights[q])
            #self.weights[q].data.div_(self.q)
        
    def reset_parameters_like_torch(self):
        #print('kaiming')
        for q in range(self.q): 
            nn.init.kaiming_uniform_(self.weights[q], a=math.sqrt(5))
            if q>0: self.weights[q].data.mul_(0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[q])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # Input to layer
        #x = torch.tanh(x)
        x = torch.cat([(x**i) for i in range(1,self.q+1)],dim=1)
        w = self.weights.transpose(0,1).reshape(self.out_channels,self.q*self.in_channels,self.kernel_size,self.kernel_size)
        x = torch.nn.functional.conv2d(x,w,bias=self.bias,padding=self.padding,dilation=self.dilation)
        return x
