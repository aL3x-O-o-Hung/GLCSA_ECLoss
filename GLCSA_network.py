import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torchvision.models as models
from torch.autograd import Variable
import torch.distributions as td
import numpy as np

device='cuda'
base_num=64

class ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,max_pool,return_single=False):
        super(ConvBlock,self).__init__()
        self.max_pool=max_pool
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.return_single=return_single
        if max_pool:
            self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        x=self.conv(x)
        b=x
        if self.max_pool:
            x=self.pool(x)
        if self.return_single:
            return x
        else:
            return x,b


class DeconvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,intermediate_channels=-1):
        super(DeconvBlock,self).__init__()
        input_channels=int(input_channels)
        output_channels=int(output_channels)
        if intermediate_channels<0:
            intermediate_channels=output_channels*2
        else:
            intermediate_channels=input_channels
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
        self.conv=ConvBlock(intermediate_channels,output_channels,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self,input_channels,num_layers,base_num):
        super(Encoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b


class Decoder(nn.Module):
    def __init__(self,num_layers,base_num):
        super(Decoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x,b):
        for i in range(self.num_layers-1):
            x=self.conv[i](x,b[i])
        return x



def custom_max(x,dim,keepdim=True):
    temp_x=x
    for i in dim:
        temp_x=torch.max(temp_x,dim=i,keepdim=True)[0]
    if not keepdim:
        temp_x=temp_x.squeeze()
    return temp_x


class GLSpatialAttentionModule(nn.Module):
    def __init__(self,num_channels):
        super(GLSpatialAttentionModule,self).__init__()
        self.conv=nn.Conv2d(in_channels=4,out_channels=1,kernel_size=(7,7),padding=3)
    def forward(self,x):
        max_x=custom_max(x,dim=[1],keepdim=True)
        avg_x=torch.mean(x,dim=1,keepdim=True)
        max_x_=custom_max(x,dim=(0,1),keepdim=True)
        avg_x_=torch.mean(x,dim=(0,1),keepdim=True)
        max_x_=max_x_.repeat(max_x.size(0),1,1,1)
        avg_x_=avg_x_.repeat(avg_x.size(0),1,1,1)
        att=torch.cat((max_x,avg_x,max_x_,avg_x_),dim=1)
        att=self.conv(att)
        att=torch.sigmoid(att)
        return x*att


class GLChannelAttentionModule(nn.Module):
    def __init__(self,in_features,im_size,reduction_rate=16):
        super(GLChannelAttentionModule,self).__init__()
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=in_features//reduction_rate))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_features//reduction_rate,out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x=custom_max(x,dim=(2,3),keepdim=False)
        avg_x=torch.mean(x,dim=(2,3),keepdim=False)
        max_x_=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        avg_x_=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        max_x_=self.linear(max_x_)
        avg_x_=self.linear(avg_x_)
        att=max_x+avg_x+max_x_+avg_x_
        att=att.unsqueeze(-1).unsqueeze(-1)
        att=torch.sigmoid(att)
        return x*att



class GLSliceAttentionModule(nn.Module):
    def __init__(self,im_size,num_channels,batch_size):
        super(GLSliceAttentionModule,self).__init__()
        weights=torch.ones(batch_size,1,1,1)
        weights=weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights=nn.Parameter(weights.clone(),True)
    def forward(self,x):
        return self.weights*x


class GLCSAModule(nn.Module):
    def __init__(self,num_slices,num_channels,im_size):
        super(GLCSAModule,self).__init__()
        self.semantic_att=GLSemanticAttentionModule(num_channels,im_size)
        self.positional_att=GLPositionalAttentionModule(num_channels)
        self.slice_att=GLSliceAttentionModule(im_size,num_channels,num_slices)
    def forward(self,x):
        x=self.semantic_att(x)
        x=self.positional_att(x)
        x=self.slice_att(x)
        return x




class EncoderGLCSA(nn.Module):
    def __init__(self,input_channels,num_layers,base_num,batch_size=20,im_size=128):
        super(EncoderGLCSA,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
        self.conv=nn.Sequential(*self.conv)
        self.attentions=[]
        for i in range(num_layers):
            self.attentions.append(GLCSAModule(batch_size,base_num*(2**i),im_size))
            im_size=im_size//2
        self.attentions=nn.Sequential(*self.attentions)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            if i!=self.num_layers-1:
                block=self.attentions[i](block)
            else:
                x=self.attentions[i](x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b



class GLCSAUNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,base_num=64,batch_size=20,im_size=128,mode='softmax'):
        super(GLCSAUNet,self).__init__()
        self.encoder=EncoderAxcroSA(input_channels,num_layers,base_num,batch_size=batch_size,im_size=im_size)
        self.decoder=Decoder(num_layers,base_num)
        self.base_num=base_num
        self.input_channels=input_channels
        self.num_classes=num_classes
        self.mode=mode
        self.conv_final=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))

    def forward(self,x):
        ##input size: 1 x input_channels x num_slices x height x width
        x=x.permute(0,2,1,3,4)
        x=x[0,:,:,:,:]
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        x=self.conv_final(x)
        if self.mode=='softmax':
            if self.num_classes==1:
                x=x.permute(1,0,2,3)
                x=x.unsqueeze(0)
                y=F.sigmoid(x)
                return x,y
            return x
        if self.mode=='dirichlet':
            x=x.permute(1,0,2,3)
            x=x.unsqueeze(0)
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            ##prob size: 1 x num_classes x num_slices x height x width
            ##uncertainty size: 1 x 1 x num_slices x height x width
            ##S size: 1 x 1 x num_slices x height x width
            ##alpha size: 1 x num_classes x num_slices x height x width
            ##evidence size: 1 x num_classes x num_slices x height x width
            return prob,uncertainty,S,alpha,evidence