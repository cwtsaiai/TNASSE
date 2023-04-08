import numpy as np

class tools():
    @staticmethod
    def DecToN(num,base):
        l = []
        if num<0:
            return "- " + DecToN(abs(num))
        else:
            while True:
                num, remainder = divmod(num,base)
                l.append(remainder)
                if num == 0:
                    return np.array(l[::-1])
    @staticmethod
    def NToDec(num,base):
        sum=0
        nump = num[::-1]
        for i in range(len(num)):
            sum += nump[i] * pow(base,i)
        return int(sum)