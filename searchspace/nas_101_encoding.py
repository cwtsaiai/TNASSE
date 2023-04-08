import numpy as np
import random
import traceback
class BACKBONE:
    def __init__(self):
        pass
    
    def get_rand_code(self):
        backbone = np.random.randint(2, size=5)
        branch = np.random.randint(2, size=21)
        operations = np.random.randint(3, size=5)
        backbone,branch = self.get_rand_backbone_branch()
        return backbone,branch,operations
    
    def get_rand_backbone_branch(self):
        backbone = np.random.randint(2, size=5)
        backbone = np.concatenate((np.ones(3),np.zeros(2)))
        random.shuffle(backbone)
        nones = np.count_nonzero(backbone == 1)
        if nones==0:
            del_zero_index = random.choice(range(5))
            backbone[del_zero_index] = 1
        
        branch = np.concatenate((np.ones(5),np.zeros(4)))
        branch = np.concatenate((branch,np.zeros(12)))
        random.shuffle(branch)
        nones = np.count_nonzero(branch == 1)
        
        #check same edges
        matrix = np.zeros([7,7])
        previous = -1
        for x,v in enumerate(backbone):
            if v==0:
                continue
            if previous==-1:
                matrix[0][x+1] = 1
            else:
                matrix[previous+1][x+1] = 1
            previous = x
        matrix[previous+1][-1] = 1
        map_backbone = matrix[np.triu_indices_from(matrix,k=1)]
        combined_m = 2*map_backbone + branch

        #check whether the edges of c1 and c2 exceeds the upper bound 9. If so, randomly delete some edges.
        zeros = np.count_nonzero(combined_m == 0)
        if zeros<12:
            del_ones_index = random.sample(list(*np.where(combined_m==1)),k=12-zeros)
            for x in del_ones_index:
                branch[x] = 0
        
        return backbone,branch

    def get_rand_backbone(self):
        backbone = np.random.randint(2, size=5)
        nones = np.count_nonzero(backbone == 1)
        if nones==0:
            del_zero_index = random.choice(range(5))
            backbone[del_zero_index] = 1
        return backbone
    
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
    
    def parse_code(self,backbone,branch,operations):
        if len(backbone)!=5 or np.count_nonzero(backbone == 1)==0:
            print('backbone: ',backbone)
            traceback.print_stack()
            raise ValueError("Invalid backbone.")
        matrix = np.zeros([7,7])
        vector = np.zeros(7)
        #Connect branch
        counter = 0
        for i in range(6,0,-1):
            for j in range(1,i+1):
                matrix[6-i][6-i+j] = branch[counter]
                counter+=1
        #Connect backbone
        previous = -1
        for i,v in enumerate(backbone):
            if v==0:
                continue
            if previous==-1:
                matrix[0][i+1] = 1
            else:
                matrix[previous+1][i+1] = 1
            previous = i
        matrix[previous+1][-1] = 1
        #Operations mapping
        vector = ['input', '' ,'' ,'' ,'' ,'' , 'output']
        for i,x in enumerate(operations):
            if x==0:
                vector[i+1] = 'conv1x1-bn-relu'
            elif x==1:
                vector[i+1] = 'conv3x3-bn-relu'
            elif x==2:
                vector[i+1] = 'maxpool3x3'
        return matrix.astype('int32'),vector

if __name__=='__main__':
    h = BACKBONE()
    c = h.get_rand_code()
    h.parse_code(*c)
