import numpy as np
import utils
import lemke
import random # random.seed
from bimatrix import payoffmatrix, uniform
from randomstart import randInSimplex, roundArray

# file format: 
# m1 m2 m3 ..... m_n denoting n player i's number of possible actions
# m1*m2 entries for A, separated by blanks / newlines
# m2*m1 entries of B, separated by blanks / newlines
# .... and so on
# blank lines or lines starting with "#" are ignored
seed = -1
trace = 100 # negative: no tracing
accuracy = 1000

class polymatrix():
    # create A given m vector
    def __init__(self, m):
        self.actions = m
        self.players = len(m)
        mainmatrix = np.zeros((self.players, self.players), dtype=object) 
        for i in range(self.players):
            for j in range(self.players):
                if i!=j:
                    mainmatrix[i][j] = utils.matrix_tofraction(np.random.randint(0, 50, [m[i], m[j]]))
        self.A = mainmatrix

    def __init__(self, M, m): #f = file containing list of matrices 

        self.actions = m
        self.players = len(m)
        mainmatrix = np.zeros( (self.players, self.players), dtype=object)
        k = 0
        for i in range(self.players):
            for j in range(self.players):
                if i!=j:
                    mainmatrix[i][j] = payoffmatrix(M[k])
                    k+=1
        self.A = mainmatrix

    def __init__(self, f): #f = file containing list of matrices 
        m, M = self.readFromFile(f)
        self.actions = m
        self.players = len(m)
        mainmatrix = np.zeros( (self.players, self.players), dtype=object)
        k = 0
        for i in range(self.players):
            for j in range(self.players):
                if i!=j:
                    mainmatrix[i][j] = payoffmatrix(M[k])
                    k+=1
        self.A = mainmatrix
    
    def __repr__(self):
        out = "# m= \n" 
        for i in self.actions:
            out += str(i) + " "
        #out += "\n # n= \n" + str(self.players)
        out += "\n# A= \n" 
        for i,r in enumerate(self.A):
            for j,c in enumerate(r):
                if i!=j: 
                    out+= " \n" + "# A" + str(i+1) + str(j+1) +" \n" + "["  +" \n"
                    out+= str(c) + " \n" +  "]"
        return out  
    
    def writeToFile(self, filename = "polygame.txt"):
        with open(filename, 'w') as f:
            f.write(repr(self))

    def readFromFile(self, filename):
        lst = utils.stripcomments(filename)
        m = []
        M = []
        for i,word in enumerate(lst):
            if i==0:
                m = [int(j) for j in word.split()]
            elif word == '[':
                x =[]
            elif word == ']':
                M.append(x)
            else:
                x.append([int(j) for j in word.split()])
        return m, M

    def createLCP(self):
        #dim for LCP = sum(m1, m2, ....) + num of players
        lcpdim = sum(self.actions) + self.players
        lcp = lemke.lcp(lcpdim)
        lcp.q[-self.players :] = [-1] * self.players 

        for i in range(self.players):
            currentrows = range(sum(self.actions[0:i+1]))[-self.actions[i]:]
            e_i = sum(self.actions)+i
            for j in range(self.players):
                currentcols = range(sum(self.actions[0:j+1]))[-self.actions[j]:]
                e_j = sum(self.actions)+j
                
                for idx_k, k in enumerate(currentrows):
                    lcp.M[k][e_i] = -1
                    for idx_l, l in enumerate(currentcols):
                            lcp.M[e_j][l] = 1
                            if self.A[i][j] != 0:
                                lcp.M[k][l] = self.A[i][j].negmatrix[idx_k][idx_l]

        # d for now
        for i in range(lcpdim):
            lcp.d[i]=1
      
        return lcp
    
    def getpriors(self, accuracy):
        priors = []
        for i in range(self.players):

            x = randInSimplex(self.actions[i])
            prior = roundArray(x, accuracy)
            priors.append(prior)
        return priors

    def runtrace(self, priors):
        lcp = self.createLCP()
        AX = []
        for i in range(self.players):
            aij_xj = [0] * self.actions[i]        # test for errors in dimensions
            for j in range(self.players):
                if i != j: # if not playing against myself
                    aij_xj = [x+y for x,y in zip(aij_xj, self.A[i][j].negmatrix @ priors[j])]
               
            AX.append(aij_xj)
        AX.append([1]*self.players)
        lcp.d = np.hstack(AX)
        tabl = lemke.tableau(lcp)
        tabl.runlemke(silent=True)
        return tuple(self.getequil(tabl))
    

    def tracing(self, trace):
        deq = {}
        if trace < 0:
            return
        elif trace == 0:
            priors = []
            for p in range(self.players):
                priors.append(uniform(self.actions[p]))
                eq = self.runtrace(priors)
                deq[eq]=1
                trace = 1 
        else:
            for k in range(trace):
                if seed >=0:
                    random.seed(10*trace*seed+k)
                priors = self.getpriors(accuracy)
                result = self.runtrace(priors)
                eq = utils.fractions_np_to_int(result)
                if eq in deq:
                    deq[eq] += 1
                else:
                    deq[eq] = 1
        self.print_statistics(deq)
        
    
    def getequil(self,tabl):
        tabl.createsol()
        return tabl.solution[1:tabl.n-(self.players-1)]
    
    def print_statistics(self, deq):
        for k in deq:
            st = ""
            start = 0
            for idx in self.actions:
                st += " ("
                for j in range(start, start+idx):
                    st += " "+ str(k[j])+" "
                st += ") "
                start += idx
            print("--------")
            print(st)
            print("found ", deq[k], " times")
            
            


    def find_supports(self, eq):
        supports = []
        idx = 0
        for i in range(self.players):
                supp = [j for j in range(self.actions[i]) if eq[idx+j]!= 0]
                supports.append(supp)
                idx += self.actions[i]
        return supports


        
#M = [[[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]]

x=polymatrix("polygame.txt")
x.tracing(trace)














