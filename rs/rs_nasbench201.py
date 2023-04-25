import numpy as np
from tqdm import tqdm
from searchspace import nas_201_encoding
encoding = nas_201_encoding.BACKBONE()

class rs():
    def __init__(self,sample,ss,ff,dictionary): 
        self.sample=sample
        self.ss = ss
        self.ff = ff
        self.savedir = 'results/'
        self.dictionary = dictionary

    def Search(self):
        numbers = np.zeros(self.sample)
        codes = np.zeros(self.sample,dtype='<S32')
        scores = np.zeros(self.sample)
        progress = tqdm(total=self.sample)
        codeslist = []
        codesdict = {}
        for i in range(self.sample):
            uid = encoding.get_rand_code()
            while str(uid) in codesdict:
                uid = encoding.get_rand_code()
            numbers[i] = i
            codes[i] = str(uid)
            codeslist.append(uid)
            codesdict[str(uid)] = uid
            scores[i] = self.ff(uid)
            progress.update(1)
            progress.set_description("Eval={};Score:{:.2f}".format(i,scores[i]))
        midx = np.argmax(scores)
        return codeslist[midx],scores[midx]
            

