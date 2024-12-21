import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import pinv, svd
from scipy.sparse.linalg import svds
from time import process_time

def proj_simplex(y):
    
    '''
    Projection onto the standard simplex
    The input y must be in the nonnegative orthant
    '''
    
    if np.sum(y) == 1:
        return y
    
    v, vv, rho = [y[0]], [], y[0]-1
    N = len(y)

    for n in range(1, N):
        yn = y[n]
        if yn > rho:
            rho += (yn-rho)/(len(v)+1)
            if rho > yn-1:
                v.append(yn)
            else:
                vv.extend(v)
                v = [yn]
                rho = yn-1

    if len(vv) > 0:
        for w in vv:
            if w > rho:
                v.append(w)
                rho += (w-rho)/len(v)

    l, flag = len(v), 1
    while flag == 1:
        for w in v:
            if w <= rho:
                v.remove(w)
                rho += (rho-w)/len(v)
        if len(v) != l:
            l, flag = len(v), 1
        else:
            flag = 0

    return np.maximum(y-rho, 0)

### l-1 ball
def LP(y):
    
    idx = np.argmax(np.abs(y))
    x = np.zeros(len(y))
    x[idx] = -np.sign(y[idx])
        
    return x

def QP(y):
    
    if np.linalg.norm(y, 1) <= 1:
        return y
    
    else:
        return np.sign(y)*proj_simplex(np.abs(y))