import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


###The code is based on https://github.com/dougbrion/pytorch-classification-uncertainty 

def KL(alpha,num_classes=2,im_size=128,batch_size=20):
    S_alpha=torch.sum(alpha,dim=1,keepdim=True)
    beta=torch.tensor(np.ones((1,num_classes,batch_size,im_size,im_size)),dtype=torch.float32)
    beta=beta.to(device)
    S_beta=torch.sum(beta,dim=1,keepdim=True)
    lnB=torch.lgamma(S_alpha)-torch.sum(torch.lgamma(alpha),dim=1,keepdim=True)
    lnB_uni=torch.sum(torch.lgamma(beta),dim=1,keepdim=True)-torch.lgamma(S_beta)
    dg0=torch.digamma(S_alpha)
    dg1=torch.digamma(alpha)
    kl=torch.sum((alpha-beta)*(dg1-dg0),dim=1,keepdim=True)+lnB+lnB_uni
    return kl


def evidential_critical_loss(prob,alpha,S,evidence,y,global_step,annealing_step=20,gamma=2,weight=30,device='cuda'):
    ##prob: output probability 1 x num_classes x num_slices x height x width
    ##alpha: dirichlet concentration 1 x num_classes x num_slices x height x width
    ##S: 1 x 1 x num_slices x height x width
    ##evidence: 1 x num_classes x num_slices x height x width
    ##true label: 1 x num_classes x num_slices x height x width
    ##global step: current epoch
    target=y[:,1,:,:,:]
    target=target.type(torch.LongTensor)
    target=target.to(device)
    A=(y-prob)**2
    B=alpha*(S-alpha)/(S*S*(S+1))
    bce_loss=F.nll_loss(torch.log(prob),target,reduction='none')
    annealing_coef=min(1,(global_step/annealing_step))
    pt=torch.exp(-bce_loss)
    weight_map=torch.ones(target.shape)
    weight_map[target==1]=weight
    weight_map=weight_map.to(device)
    D=((1-pt)**gamma)*(A+B)*weight_map
    alp=evidence*(1-y)+1
    C=annealing_coef*KL(alp)
    return torch.mean(D+C)