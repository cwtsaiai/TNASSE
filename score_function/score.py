import torch
import numpy as np
from numpy.linalg import norm
import copy
import sys
sys.path.append('../score_function')
sys.path.append('./score_function')
from utils import add_dropout, init_network 
from Product import score as Product
import re
import counter 


def score(code,get_net_by_code,train_loader,device,args,data,target,noise):
    try:
        network,_ = get_net_by_code(code)
    except:
        try:
            network = get_net_by_code(code)
        except Exception as e:
            return -1e8

    noise_layers = 0
    data_layers = 0
    n1 = copy.deepcopy(network)
    n2 = copy.deepcopy(network)
    pack = (copy.copy(data), copy.copy(target), copy.copy(noise))
    noise_layers,n_conv,channel= Product(n1, pack, device, args)

    pack = (copy.copy(data), copy.copy(target))
    data_layers,_,_= Product(n2, pack, device, args)
    errs = []
    for i in range(len(noise_layers)):
        error = noise_layers[i] - data_layers[i]
        errs.append(np.sum(np.square(error))/error.size)
    try:
        epsilon=1e-10
        theta = 0

        eta = np.log(epsilon+np.sum(errs))
        gamma = channel
        rho = n_conv/len(errs)
        
        if eta>theta:
            Psi = np.log((gamma*rho)/eta)
        else:
            Psi = 0
        
    except Exception as e:
        Psi=0

    del(noise_layers)
    del(data_layers)
        
    try:
        counter.EVALS+=1
    except:
        pass
    del(n1)
    del(n2)
    return Psi