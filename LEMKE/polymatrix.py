
import sys
import numpy as np
import math
import fractions
import utils
import columnprint
import lemke
import randomstart 
import random # random.seed
from bimatrix import payoffmatrix, getequil, str_eq, supports, rangesplit
from randomstart import randInSimplex

# file format: 
# m1 m2 m3 ..... m_n denoting n player i's number of possible actions
# m1*m2 entries for A, separated by blanks / newlines
# m2*m1 entries of B, separated by blanks / newlines
# .... and so on
# blank lines or lines starting with "#" are ignored


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
    
    def runLH(self, droppedlabel):
        lcp = self.createLCP()
        lcp.d[droppedlabel-1] = 0  # subsidize this label
        tabl = lemke.tableau(lcp)
        # tabl.runlemke(verbose=True, lexstats=True, z0=gz0)
        tabl.runlemke(silent=False)
        return tuple(getequil(tabl))
    
    def LH(self, LHstring):
        if LHstring == "":
            return
        m = sum(self.actions)
        n = self.players
        lhset = {} # dict of equilibria and list by which label found
        labels = rangesplit(LHstring, m+n)
        # how many labels? what to assign to LHstring?
    



M = [[[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]]

x = polymatrix("polygame1.txt")
print(x)
y = x.createLCP()
print(y)


















# Archives - code written for testing, not used at the moment.
class sampleMatrix():
    def __init__(self, m): #m = [m1, m2, m3, ....., m_n]
        self.m = m
        self.n = len(m)
        self.A = self.getRandomPayoffs(m)

    def getRandomPayoffs(m):
        numplayers = len(m)
        A = np.zeros( (numplayers, numplayers), dtype=object) 
        for i in range(numplayers):
            for j in range(numplayers):
                if i != j:
                    matrix = np.random.rand(m[i], m[j])
                    matrix = utils.matrix_tofraction(matrix)
                    A[i][j] = matrix
        return A
    
    def __init__(self, m, Matrices): #Matrices = list of matrices sorted by the indices of row player asc
        self.m = m
        self.n = len(m)
        mainmatrix = np.zeros( (self.n, self.n), dtype=object) 
        k=0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    mainmatrix[i][j] = utils.matrix_tofraction(Matrices[k])
                    k+=1
        self.A = mainmatrix


