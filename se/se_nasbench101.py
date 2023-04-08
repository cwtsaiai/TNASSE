import math
import numpy as np 
import random
import time
from tqdm import trange
from tqdm import tqdm
from .tools import *
DecToN = tools.tools.DecToN
NToDec = tools.tools.NToDec
import copy
from searchspace import nas_101_encoding
encoding = nas_101_encoding.BACKBONE()
from treelib import Tree, Node
import counter

from scipy.stats import truncnorm
scale = 3.
rang = 1

class se():
    """
    NASBench101 version
    Search Economics (SE) --basic version
    Python implementation
    Note: In this version, n=h is assumed.

    $$$$$$$$$$$$$$$ Parameters $$$$$$$$$$$$$$$$$$$$
        iters --> number of iterations for each run
        n --> number of searchers
        h --> number of regions
        w --> the number of possible goods 
              (the number of samples of each region)
        sl --> sequence length
        ff --> fitness function
        atom --> base
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    """
    def __init__(self,iters,n,h,w,sl,atom,max_nodes,ff,dictionary,max_evaluations,good_init_code=None):
        self.iters = iters
        self.n = n
        self.h = h 
        self.w = w
        self.sl = sl
        self.dictionary = dictionary
        self.ff = ff
        self.atom = atom
        self.good_init_code = good_init_code
        self.max_nodes = max_nodes
        self.max_evaluations = max_evaluations
        counter.eval_init()

    def Search(self):
        self.region = {}
        gbest = -1e8
        gbest_code = -np.ones(self.sl)
        s_backbone,s_m,s_v = self.Initialization()
        s_backbone,s_m,s_v,m_backbone,m_m,m_v = self.ResourceArrangement(s_backbone,s_m,s_v,self.region)
        progress = tqdm(total=self.max_evaluations)
        i=1
        prev_eval = 0
        while counter.EVALS<self.max_evaluations:
            if counter.EVALS>self.max_evaluations:
                break
            s = self.VisionSearch((s_backbone,s_m,s_v),(m_backbone,m_m,m_v))
            cbest_code = self.mvmax(s[0],s[1],s[2],self.ff)
            cbest = self.ff(cbest_code)
            if cbest > gbest:
                gbest_code = cbest_code
                gbest = cbest
            m = self.MarketingResearch((s[0],s[1],s[2]),(m_backbone,m_m,m_v))
            progress.update(counter.EVALS-prev_eval)
            progress.set_description("Eval={};Iter={};gbest:{:.2f}".format(counter.EVALS,i,gbest))
            prev_eval = counter.EVALS
            i+=1
        return gbest_code, gbest

    def Initialization(self):
        self.ta = np.ones(self.h)
        self.tb = np.ones(self.h)
        return np.zeros((self.n,self.max_nodes),dtype=int),np.zeros((self.n,self.sl),dtype=int),np.zeros((self.n,self.max_nodes),dtype=int)

    def mvmax(self,backbone,m,v,f):
        max_value = -1e10
        if len(backbone) != len(m) != len(v):
            raise("The length of backbone doesn't equal to the length of m and v.")
        for i in range(len(m)):
            temp_value = f((backbone[i],m[i],v[i]))
            if temp_value>max_value:
                max_value = temp_value
                max_backbone = backbone[i]
                max_m = m[i]
                max_v = v[i]
        return max_backbone,max_m,max_v

    def ResourceArrangement(self,s_backbone,s_m,s_v,region):
        self.nfix = math.ceil(round(math.log(self.h,self.atom),6))
        #check edge case (h==1)
        if self.nfix==0:
            self.nfix = 1
        divided_regions = np.full((self.atom),self.h//self.atom)
        index = random.sample(list(np.arange(self.atom)),self.h%self.atom)
        divided_regions[index]+=1
        ################################
        tree = Tree()
        branch=1
        x = 0
        tree.create_node('root','root')
        parent_node = 'root'
        def isPower(num,base):
            if base in {0,1}:
                return num==base
            testnum = base
            while testnum<num:
                testnum = testnum*base 
            return testnum==num
        if isPower(self.h,self.atom):
            for i in range(self.nfix):
                for j in range(branch):
                    for k in range(self.atom):
                        tree.create_node(k,x,parent=parent_node)
                        x+=1
                    parent_node = x//self.atom-1
                branch*=self.atom
            all_paths = tree.paths_to_leaves()
            for i in range(self.h): 
                region[i] = [tree[id].tag for id in all_paths[i%self.atom**math.floor(math.log(self.h,self.atom))][1:]]
        else:
            for i in range(self.nfix-1):
                for j in range(branch):
                    for k in range(self.atom):
                        tree.create_node(k,x,parent=parent_node)
                        x+=1
                    parent_node = x//self.atom-1
                branch*=self.atom
            all_paths = tree.paths_to_leaves()
            for i in range(self.h): 
                region[i] = [tree[id].tag for id in all_paths[i%self.atom**math.floor(math.log(self.h,self.atom))][1:]]

            c = 0
            for i in range(self.h-self.atom**(self.nfix-1),self.h):
                ndiv = math.ceil((i+1)/(self.atom**(self.nfix-1)))
                group = np.array_split(np.arange(self.atom),ndiv)
                for cc,j in enumerate(group):
                    region[i%self.atom**(self.nfix-1)+cc*self.atom**(self.nfix-1)].append(list(j))

        m_backbone = np.zeros((self.h,self.w,self.max_nodes),dtype=int)
        m_m = np.zeros((self.h,self.w,self.sl),dtype=int)
        m_v = np.zeros((self.h,self.w,self.max_nodes),dtype=int)
        self.rb_backbone = -np.ones((self.h,self.max_nodes))
        self.rb_m = -np.ones((self.h,self.sl))
        self.rb_v = -np.ones((self.h,self.max_nodes))
        
        for j in range(self.h):
            for k in range(self.w):
                m_backbone[j,k],m_m[j,k] = encoding.get_rand_backbone_branch()
                m_v[j,k] = np.concatenate((self.get_value(j),np.random.randint(0,self.atom,size=self.max_nodes-self.nfix)))

            self.rb_backbone[j],self.rb_m[j],self.rb_v[j] = self.mvmax(m_backbone[j],m_m[j],m_v[j],self.ff)
        
        for i in range(self.n):
            #Here, si is just assigned by "randomly" selecting from the goods.
            s_backbone[i] = m_backbone[i%self.h,random.randint(0,self.w-1)]
            s_m[i] = m_m[i%self.h,random.randint(0,self.w-1)]
            s_v[i] = m_v[i%self.h,random.randint(0,self.w-1)]
        return s_backbone,s_m,s_v,m_backbone,m_m,m_v
        

    def VisionSearch(self,s,m):
        self.v = self.Transition(s,m)
        e = self.ExpectedValue(self.v,m,self.ta,self.tb)
        s = self.Determination(self.v,e)
        return s

    def MarketingResearch(self,s,m):
        m = self.Update(s,m)
        self.Accumulation1(s,m)
        self.Accumulation1(s,m)
        return m

    def Transition(self,s,m):
        s_backbone,s_m,s_v = copy.deepcopy(s[0]),copy.deepcopy(s[1]),s[2] 
        m_backbone,m_m,m_v = copy.deepcopy(m[0]),copy.deepcopy(m[1]),m[2]
        v_backbone = np.zeros((self.n,self.h,self.w,self.max_nodes),dtype=int)
        v_m = np.zeros((self.n,self.h,self.w,self.sl),dtype=int)
        v_v = np.zeros((self.n,self.h,self.w,self.max_nodes),dtype=int)
        for i in range(self.n):
            for j in range(self.h):
                for k in range(self.w):
                    cp = random.randint(1,self.max_nodes-2)
                    c1_backbone = np.concatenate((s_backbone[i][:cp], m_backbone[j][k][cp:]))
                    c2_backbone = np.concatenate((m_backbone[j][k][:cp],s_backbone[i][cp:]))
                    mp = random.randint(0,self.max_nodes-1) 
                    c1_backbone[mp] = 1 ^ c1_backbone[mp]
                    c2_backbone[mp] = 1 ^ c2_backbone[mp]

                    c1_nones = np.count_nonzero(c1_backbone[i] == 1)
                    c2_nones = np.count_nonzero(c2_backbone[i] == 1)
                    if c1_nones==0:
                        del_zero_index = random.choice(range(5))
                        c1_backbone[del_zero_index] = 1
                    if c2_nones==0:
                        del_zero_index = random.choice(range(5))
                        c2_backbone[del_zero_index] = 1

                    cp = random.randint(1,self.sl-2)
                    c1_m = np.concatenate((s_m[i][:cp], m_m[j][k][cp:]))
                    c2_m = np.concatenate((m_m[j][k][:cp],s_m[i][cp:]))

                    #one-point mutation
                    mp = random.randint(0,self.sl-1) 
                    c1_m[mp] = 1 ^ c1_m[mp]
                    c2_m[mp] = 1 ^ c2_m[mp]

                    #check same edges
                    matrix = np.zeros([7,7])
                    previous = -1
                    for x,v in enumerate(c1_backbone):
                        if v==0:
                            continue
                        if previous==-1:
                            matrix[0][x+1] = 1
                        else:
                            matrix[previous+1][x+1] = 1
                        previous = x
                    matrix[previous+1][-1] = 1
                    map_backbone = matrix[np.triu_indices_from(matrix,k=1)]
                    combined_m_1 = 2*map_backbone + c1_m

                    matrix = np.zeros([7,7])
                    previous = -1
                    for x,v in enumerate(c2_backbone):
                        if v==0:
                            continue
                        if previous==-1:
                            matrix[0][x+1] = 1
                        else:
                            matrix[previous+1][x+1] = 1
                        previous = x
                    matrix[previous+1][-1] = 1
                    map_backbone = matrix[np.triu_indices_from(matrix,k=1)]
                    combined_m_2 = 2*map_backbone + c2_m

                    #check whether the edges of c1 and c2 exceeds the upper bound 9. If so, randomly delete some edges.
                    c1_zeros = np.count_nonzero(combined_m_1 == 0)
                    c2_zeros = np.count_nonzero(combined_m_2 == 0)
                    if c1_zeros<12:
                        del_ones_index_1 = random.sample(list(*np.where(combined_m_1==1)),k=12-c1_zeros)
                        for x in del_ones_index_1:
                            c1_m[x] = 0
                    if c2_zeros<12:
                        del_ones_index_2 = random.sample(list(*np.where(combined_m_2==1)),k=12-c2_zeros)
                        for x in del_ones_index_2:
                            c2_m[x] = 0

                    #random one-point crossover (s_v,m_v)
                    cp = random.randint(self.nfix,self.max_nodes-2)
                    c1_v = np.concatenate((s_v[i][:self.nfix],s_v[i][self.nfix:cp], m_v[j][k][cp:]))
                    c2_v = np.concatenate((s_v[i][:self.nfix],m_v[j][k][self.nfix:cp],s_v[i][cp:]))
                    #one-point mutation
                    mp = random.randint(self.nfix,self.max_nodes-1) 
                    c1_v[mp] = random.choice(list(set([i for i in range(self.atom)])-set([c1_v[mp]])))
                    c2_v[mp] = random.choice(list(set([i for i in range(self.atom)])-set([c2_v[mp]])))

                    if self.ff((c1_backbone,c1_m,c1_v))>self.ff((c2_backbone,c2_m,c2_v)):
                        v_backbone[i][j][k] = c1_backbone
                        v_m[i][j][k] = c1_m
                        v_v[i][j][k] = c1_v
                    else:
                        v_backbone[i][j][k] = c2_backbone
                        v_m[i][j][k] = c2_m
                        v_v[i][j][k] = c2_v
        return v_backbone,v_m,v_v           

    def ExpectedValue(self,v,m,ta,tb):
        e = np.zeros((self.n,self.h))
        for i in range(self.n):
            for j in range(self.h):
                #e^i_j = T_j*V^i_j*M_j
                Tj = tb[j]/ta[j]
                Vij = sum( [self.ff((v[0][i,j,k],v[1][i,j,k],v[2][i,j,k])) for k in range(self.w)] ) / self.w
                Mj = self.ff((self.rb_backbone[j],self.rb_m[j],self.rb_v[j])) / sum([self.ff((m[0][j,k],m[1][j,k],m[2][j,k])) for k in range(self.w)])
                e[i][j] = Tj * Vij * Mj
        return e

    def Determination(self,v,e):
        if self.w<=2:
            k = self.w
        else:
            k = math.ceil(self.w/2)
        v_backbone = v[0]
        v_m = v[1]
        v_v = v[2]
        s_backbone = np.zeros((self.n,self.max_nodes),dtype=int)
        s_m = np.zeros((self.n,self.sl),dtype=int)
        s_v = np.zeros((self.n,self.max_nodes),dtype=int)
        for i in range(self.n):
            # Can be modified by using tournament selection. But there is only 2 goods in each region for this moment
            ebarg = np.argmax(e[i]) 
            cand_ids = random.sample(range(self.w),k)
            cand_backbone = copy.deepcopy(v[0][i][ebarg][cand_ids])
            cand_m = copy.deepcopy(v[1][i][ebarg][cand_ids])
            cand_v = copy.deepcopy(v[2][i][ebarg][cand_ids])
            max_cand_score = float('-inf')
            for j in range(k):
                cand_v[j][:self.nfix] = self.get_value(ebarg)
                new_cand_score = self.ff((cand_backbone[j],cand_m[j],cand_v[j]))
                if new_cand_score>max_cand_score:
                    max_cand_backbone,max_cand_m,max_cand_v = cand_backbone[j],cand_m[j],cand_v[j]
                    max_cand_score = new_cand_score
            s_backbone[i] = max_cand_backbone
            s_m[i] = max_cand_m
            s_v[i] = max_cand_v
        return s_backbone,s_m,s_v

    def Update(self,s,m):
        s_backbone,s_m,s_v = s[0],s[1],s[2]
        m_backbone,m_m,m_v = m[0],m[1],m[2]
        for i in range(self.n):
            # Region of searcher i
            sreg = self.get_key(s_v[i])
            fm = [self.ff((m_backbone[sreg][x],m_m[sreg][x],m_v[sreg][x])) for x in range(len(m_v[sreg]))]
            marg = np.argmin(fm)
            mmin = np.min(fm)
            if self.ff((s_backbone[i],s_m[i],s_v[i])) > mmin:
                m_backbone[sreg][marg] = s_backbone[i]
                m_m[sreg][marg] = s_m[i]
                m_v[sreg][marg] = s_v[i]
        return m_backbone,m_m,m_v

    def get_key(self,s):
        for k,v in self.region.items():
            if type(v[-1])==int and s[self.nfix-1]==v[-1]:
                return k
            if type(v[-1])!=int and list(s[:self.nfix-1])==v[:-1] and s[self.nfix-1] in v[-1]:
                return k

    def get_value(self,index):
        if not isinstance(self.region[index][-1],list):
            y = [self.region[index][-1]]
        else:
            y = self.region[index][-1]
        return np.concatenate((np.array(self.region[index][:-1]),np.array(random.sample(y,1))))

    def Accumulation1(self,s,m):
        for i in range(self.n):
            self.ta[self.get_key(s[2][i])]+=1

    def Accumulation2(self,s,m):
        self.tb+=1
        for i in range(self.n):
            self.tb[self.get_key(s[2][i])]=1
