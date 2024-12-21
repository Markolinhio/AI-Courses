#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:18:25 2019

@author: gionuno
"""

import numpy        as np;
import numpy.random as rd;
import torch;

import matplotlib.colors as csp;
import matplotlib.pyplot as plt;
import matplotlib.image  as img;

def proj_splx(y):
    sy = np.sort(y)[::-1];
    cy = np.cumsum(sy);
    t = None;
    for i in range(y.shape[0]-1):
        t_i = (cy[i]-1.0)/(i+1);
        if t_i >= sy[i+1]: 
            t = t_i; 
            break;
    if t == None:
        t = (np.sum(y)-1.0)/y.shape[0];
    x = (y > t)*(y-t);
    return x;

smallest_float = np.nextafter(np.float32(0), np.float32(1))
float_type = np.longdouble

def max_norm(x):
    return np.amax(np.abs(x))

def get_entropy(P):
    logP = torch.log(P + 1e-20)
    return -1 * torch.sum(logP * P - P)

def get_KL(P, Q):
    log_ratio = torch.log(P + 1e-20) - torch.log(Q + 1e-20)
    return torch.sum(P * log_ratio - P + Q)

def compute_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)


def compute_KL(P, Q):
    log_ratio = np.log(P) - np.log(Q)
    return np.sum(P * log_ratio - P + Q)


def dot(x, y):
    return np.sum(x * y)

def dotp(x, y):
    return torch.sum(x * y)

def norm1(X):
    return torch.sum(torch.abs(X))

def supnorm(X):
    return torch.max(torch.abs(X))

#--- Sparse OT computation

def entropic_map(alpha,beta,C,igam):
    return np.exp((np.outer(alpha,np.ones(beta.shape))+np.outer(np.ones(alpha.shape),beta)-C)/igam)

def entropic(a,b,igam,T,alpha,beta):
    return 0

def opt_entropic(a,b,igam,T,alpha,beta):
    alpha = 1e-14*rd.rand(a.shape[0]);
    beta  = 1e-14*rd.rand(b.shape[0]);
    
    galphal = np.zeros(alpha.shape);
    gbetal  = np.zeros(beta.shape);
    
    salpha  = np.zeros(alpha.shape);
    sbeta   = np.zeros(beta.shape);
    
    T = entropic_map(alpha,beta,C);

    otab = entropic(a,b,igam,T,alpha,beta);
    
    c1 = 1e-4; 
    
    dt = 1.0;
    
    for l in range(L+1):
        pass
    
    return 0

def grad_entropic_alpha(a,igam,T):
    return a - igam*np.sum(T,axis=1);

def grad_entropic_beta(b,igam,T):
    return b - igam*np.sum(T,axis=0);

def ot_map(alpha,beta,C): # alpha, beta here act as u, v
    T = np.outer(alpha,np.ones(beta.shape))+np.outer(np.ones(alpha.shape),beta)-C; # max(alpha*1 + 1*beta - C, 0)
    T = T*(T>0.0);
    return T;

def ot(a,b,igam,T,alpha,beta):
    return np.dot(a,alpha)+np.dot(b,beta)-0.5*igam*np.sum(T**2);
    
def grad_ot_alpha(a,igam,T):
    return a - igam*np.sum(T,axis=1);

def grad_ot_beta(b,igam,T):
    return b - igam*np.sum(T,axis=0);

def opt_ot(a,b,C,igam=1e-1,L=100):
    
    alpha = 1e-14*rd.rand(a.shape[0]);
    beta  = 1e-14*rd.rand(b.shape[0]);
    
    galphal = np.zeros(alpha.shape);
    gbetal  = np.zeros(beta.shape);
    
    salpha  = np.zeros(alpha.shape);
    sbeta   = np.zeros(beta.shape);
    
    T = ot_map(alpha,beta,C);

    otab = ot(a,b,igam,T,alpha,beta);
    
    c1 = 1e-4; 
    
    dt = 1.0;
    
    for l in range(L+1):
        dt = min(1.25*dt,1.0);
        
        galpha = grad_ot_alpha(a,igam,T);
        
        balpha = 0.0 if l == 0 else np.dot(galpha,galpha-galphal)/(1e-14+np.dot(galphal,galphal));
        
        salpha = galpha + balpha*salpha; 
        
        next_T      = ot_map(alpha+dt*salpha,beta,C);
        next_otab   = ot(a,b,igam,next_T,alpha+dt*salpha,beta);
        
        dtg_alpha = np.dot(galpha,salpha);
        
        for k in range(10):
            if next_otab >= otab + c1*dt*dtg_alpha:
                break;
            dt *= 0.75;
            next_T      = ot_map(alpha+dt*salpha,beta,C);
            next_otab   = ot(a,b,igam,next_T,alpha+dt*salpha,beta);
                
        alpha = alpha+dt*salpha;
        
        galphal = galpha;
        
        T    = next_T;
        otab = next_otab;

        dt = min(1.25*dt,1.0);
        
        gbeta  = grad_ot_beta(b,igam,T);

        bbeta = 0 if l == 0 else np.dot(gbeta,gbeta-gbetal)/(1e-14+np.dot(gbetal,gbetal));
        
        sbeta = gbeta + bbeta*sbeta;
        
        next_T     = ot_map(alpha,beta+dt*sbeta,C);
        next_otab  = ot(a,b,igam,next_T,alpha,beta+dt*sbeta); 

        dtg_beta = np.dot(gbeta,sbeta);
        
        for k in range(10):
            if next_otab >= otab + c1*dt*dtg_beta:
                break;
            dt *= 0.75;
            next_T     = ot_map(alpha,beta+dt*sbeta,C);
            next_otab  = ot(a,b,igam,next_T,alpha,beta+dt*sbeta);
        
        beta = beta+dt*sbeta;
        
        gbetal = gbeta;
        
        T    = next_T;
        otab = next_otab;
        if l%10 == 0:
            print(str(l)+" "+str(otab)+" "+str(np.sum(C*T)/(np.sum(T)+1e-14))+" "+str(dt));
    return T;
