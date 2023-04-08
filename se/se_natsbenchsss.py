import math
import numpy as np 
import random
from tqdm import trange
from tqdm import tqdm
from .tools import *
DecToN = tools.tools.DecToN
NToDec = tools.tools.NToDec
from treelib import Tree, Node
import counter

class se():
    """
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
    def __init__(self,iters,n,h,w,sl,atom,ff,dictionary,max_evaluations,good_init_code=None):
        self.iters = iters
        self.n = n
        self.h = h 
        self.w = w
        self.sl = sl
        self.dictionary = dictionary
        self.ff = ff
        self.max_evaluations = max_evaluations
        self.atom = atom
        self.good_init_code = good_init_code
        counter.eval_init()

    def Search(self):
        self.r = []
        gbest = float('-inf')
        gbest_code = -np.ones(self.sl)
        s = self.Initialization()
        s,m = self.ResourceArrangement(s,self.r)
        progress = tqdm(total=self.max_evaluations)
        i=1
        prev_eval = 0
        while counter.EVALS<self.max_evaluations:
            if counter.EVALS>self.max_evaluations:
                break
            s = self.VisionSearch(s,m)
            cbest_code = max(s,key=self.ff)
            gbest_code = cbest_code if self.ff(cbest_code) > gbest else gbest_code
            gbest = self.ff(gbest_code)
            m = self.MarketingResearch(s,m)
            progress.update(counter.EVALS-prev_eval)
            progress.set_description("Eval={};Iter={};gbest:{:.2f}".format(counter.EVALS,i,gbest))
            prev_eval = counter.EVALS
            i+=1
        return gbest_code, gbest
    def Initialization(self):
        self.ta = np.ones(self.h)
        self.tb = np.ones(self.h)
        return np.zeros((self.n,self.sl))
    
    def ResourceArrangement(self,s,r):
        self.region = {}
        self.nfix = math.ceil(round(math.log(self.h,self.atom),6))
        #check edge case (h==1)
        if self.nfix==0:
            self.nfix = 1
        divided_regions = np.full((self.atom),self.h//self.atom)
        index = random.sample(list(np.arange(self.atom)),self.h%self.atom)
        divided_regions[index]+=1
        
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
                self.region[i] = [tree[id].tag for id in all_paths[i%self.atom**math.floor(math.log(self.h,self.atom))][1:]]
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
                self.region[i] = [tree[id].tag for id in all_paths[i%self.atom**math.floor(math.log(self.h,self.atom))][1:]]

            c = 0
            for i in range(self.h-self.atom**(self.nfix-1),self.h):
                ndiv = math.ceil((i+1)/(self.atom**(self.nfix-1)))
                group = np.array_split(np.arange(self.atom),ndiv)
                for cc,j in enumerate(group):
                    self.region[i%self.atom**(self.nfix-1)+cc*self.atom**(self.nfix-1)].append(list(j))

        m = np.zeros((self.h,self.w,self.sl))
        self.rb = -np.ones((self.h,self.sl))
        
        for j in range(self.h):
            for k in range(self.w):
                m[j,k] = np.concatenate((self.get_value(j),np.random.randint(0,self.atom,size=self.sl-self.nfix)))
            self.rb[j] = max(m[j],key = self.ff)
        for i in range(self.n):
            #Here, si is just assigned by "randomly" selecting from the goods.
            s[i] = m[i%self.h,random.randint(0,self.w-1)]
        return s,m

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
        v = np.zeros((self.n,self.h,self.w,self.sl))
        for i in range(self.n):
            for j in range(self.h):
                for k in range(self.w):
                    #random one-point crossover
                    cp = random.randint(self.nfix,self.sl-2)
                    c1 = np.concatenate((s[i][:self.nfix],s[i][self.nfix:cp], m[j][k][cp:]))
                    c2 = np.concatenate((s[i][:self.nfix],m[j][k][self.nfix:cp],s[i][cp:]))

                    #one-point mutation
                    mp = random.randint(self.nfix,self.sl-1) 
                    c1[mp] = random.choice(list(set([i for i in range(self.atom)])-set([c1[mp]])))
                    c2[mp] = random.choice(list(set([i for i in range(self.atom)])-set([c2[mp]])))
                    v[i][j][k] = c1 if self.ff(c1)>self.ff(c2) else c2
        return v           

    def ExpectedValue(self,v,m,ta,tb):
        e = np.zeros((self.n,self.h))
        for i in range(self.n):
            for j in range(self.h):
                #e^i_j = T_j*V^i_j*M_j
                Tj = tb[j]/ta[j]
                Vij = sum( [self.ff(v[i][j][k]) for k in range(self.w)] ) / self.w
                Mj = self.ff(self.rb[j]) / sum([self.ff(m[j][k]) for k in range(self.w)])
                e[i][j] = Tj * Vij * Mj
        return e

    def Determination(self,v,e):
        if self.w<=2:
            k = self.w
        else:
            k = math.ceil(self.w/2)
        s = np.zeros((self.n,self.sl))
        for i in range(self.n):
            # Can be modified by using tournament selection.
            ebarg = np.argmax(e[i])
            cand = np.array(random.sample(list(v[i][ebarg]),k))
            for j in range(len(cand)):
                cand[j][:self.nfix] = self.get_value(ebarg)
            s[i] = max(cand,key=self.ff)
        return s

    def Update(self,s,m):
        for i in range(self.n):
            # Region of searcher i
            sreg = self.get_key(s[i])
            fm = [self.ff(x) for x in m[sreg]]
            marg = np.argmin(fm)
            mmin = np.min(fm)
            if self.ff(s[i]) > mmin:
                m[sreg][marg] = s[i]
        return m

    def get_key(self,s):
        for k,v in self.region.items():
            if list(s[:self.nfix-1])==v[:-1] and s[self.nfix-1] in v[-1]:
                return k

    def get_value(self,index):
        if not isinstance(self.region[index][-1],list):
            y = [self.region[index][-1]]
        else:
            y = self.region[index][-1]
        return np.concatenate((np.array(self.region[index][:-1]),np.array(random.sample(y,1))))

    def Accumulation1(self,s,m):
        for i in range(self.n):
            self.ta[self.get_key(s[i])]+=1

    def Accumulation2(self,s,m):
        self.tb+=1
        for i in range(self.n):
            self.tb[self.get_key(s[i])]=1
    
