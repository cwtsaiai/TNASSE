from ast import Pass
import numpy as np
import random

class BACKBONE:
    def __init__(self):
        pass
    
    def get_rand_code(self):
        return np.concatenate((np.random.randint(5,size=6),np.random.randint(4,size=1),np.random.randint(1,5,size=3)))
    
    def get_rand_backbone_branch(self):
        pass

    def get_rand_backbone(self):
        pass
    
    def get_rand_branch(self):
        branch = np.random.randint(2, size=21)
        nones = np.count_nonzero(branch == 1)
        if nones>9:
            ones_index = np.where(branch==1)
            del_ones_index = random.sample(list(*ones_index),k=nones-9)
            for x in del_ones_index:
                branch[x] = 0
        return branch
    
    def get_rand_operations(self):
        return np.random.randint(3, size=5)
    
    def parse_code(self,code):
        newcode = code[:6]
        mapping = {'a':0,'b':2,'c':5,'d':1,'e':3,'f':4}
        backbone_choices = {0:np.array(['c','d','z']),1:np.array(['e','z','z']),2:np.array(['a','b','c']),3:np.array(['a','f','z'])}
        for n,i in enumerate(backbone_choices[code[6]]):
            if i not in mapping:
                break
            newcode[mapping[i]] = code[6+1+n]
        return newcode

if __name__=='__main__':
    h = BACKBONE()
    c = np.array([0,1,2,3,0,1,2,2,3,0])
    c = np.array([0,1,4,3,4,0,0,0,1,-1])
    print(h.parse_code(c))
