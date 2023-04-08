import numpy as np
import random
import traceback
class ENCODING:
    def __init__(self):
        self.candidates = [8, 16, 24, 32, 40, 48, 56, 64]
        self.num_choices = 8
    def __len__(self):
        return 5
    def get_rand_code(self):
        return np.array(random.sample(range(0,self.num_choices),k=5))

    def parse_code(self,code):
        if type(code)==np.int64 or type(code)==int:
            return int(code)
        if type(code)==np.ndarray and type(code[0])==np.float64:
            code = code.astype(int)
        if type(code)==list and type(code[0])==float:
            code = [int(x) for x in code]
        newcode=[]
        for c in code:
            newcode.append(self.candidates[c])
        index = 0
        for i,c in enumerate(newcode):
            index+=self.candidates.index(c)*pow(self.num_choices,len(self)-1-i)
        return index
    def encode_index(self,index):
        code = np.zeros(len(self))
        i=0
        while index > 0:
            code[len(self)-1-i] = index%self.num_choices
            index//=self.num_choices
            i+=1
        return code

if __name__=='__main__':
    e = ENCODING()
    code = [1,1,1,1,1]
    print(e.parse_code(code))
    print(e.encode_index(e.parse_code(code)))