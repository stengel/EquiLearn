#!/usr/bin/python
# bimatrix class

import sys
import numpy as np
import fractions
import utils
import columnprint
import lemke
import randomstart 
import random # random.seed

# for debugging
def printglobals(): 
    globs = [x for x in globals().keys() if not "__" in x]
    for var in globs:
        value = str(globals()[var])
        if not "<" in value:
            print("    "+str(var)+"=", value)

# file format: 
# <m> <n>
# m*n entries of A, separated by blanks / newlines
# m*n entries of B, separated by blanks / newlines
#
# blank lines or lines starting with "#" are ignored

# defaults
# MAXDIM = 2000 # largest allowed value for m and n; not used yet
gamefilename = "game"
gz0 = False
LHstring = "" # empty means LH not called
seed = -1
trace = -1 # negative: no tracing
# accuracy = DEFAULT_accuracy = 1000
accuracy = 1000

# amends defaults
def processArguments():
    global gamefilename,gz0,LHstring,seed,trace,accuracy
    arglist = sys.argv[1:]
    setLH = False
    settrace = False
    setaccuracy = False
    setseed = False
    setdecimals = False
    showhelp = False
    for s in arglist:
        if s[0] == '-': 
            # discard optional argument-parameters
            if (setLH or settrace or setseed):
                setLH = settrace = setseed = False
            if s=="-LH":
                setLH = True
                LHstring = "1-"
            elif s=="-trace":
                settrace = True
                trace = 0
            elif s=="-seed":
                setseed = True
                seed = 0
            elif s=="-accuracy":
                setaccuracy = True
            elif s=="-decimals":
                setdecimals = True
            elif s=="-z0":
                gz0 = True
            elif s=="-?" or s=="-help":
                showhelp = True
            else: # any other "-" argument
                showhelp = True
                print("! unknown option: ", repr(s))
        else: # s not starting with "-"
            if setLH:
                setLH = False
                LHstring = s
            elif settrace:
                settrace= False
                trace = int(s)
            elif setseed:
                setseed= False
                seed = int(s)
            elif setdecimals:
                setdecimals = False
                utils.setdecimals(int(s))
            elif setaccuracy:
                setaccuracy= False
                accuracy = int(s)
            else: 
                gamefilename = s
    if (showhelp):
        helpstring="""usage: bimatrix.py [options]
options:
    <filename>      here: """+repr(gamefilename)+ """, must not start with '-'
    -LH [<range>] : Lemke-Howson with missing labels, e.g. '1,3-5,7-' ('' = all)
    -trace [<num>]: tracing procedure, <num> no. of priors, 0 = centroid
    -seed [<num>] : random seed, default: None 
    -accuracy <n> : accuracy prior, <n>=denominator, here """+str(accuracy)+"""
    -decimals <d> : allowed payoff digits in input after decimal point, default 4
    -?, -help:      show this help and exit"""
        print(helpstring)
        exit(0)
    return

# list generated from string s such as "1-3,10,4-7", all not
# larger than endrange (50 is arbitrary default) and at least 1
def rangesplit(s,endrange=50): 
    result = []
    for part in s.split(','):
        if part != "":
            if '-' in part:
                a, b = part.split('-')
                a = int(a) 
                b = endrange if b=="" else int(b) 
            else: 
                a = int(part)
                b = a
            a = max(a,1)
            b = min(b, endrange) # a > endrange means empty range
            result.extend(range(a, b+1))
    return result 

# used for both A and B
class payoffmatrix:
    # create zero matrix of given dimensions
    def __init__(self, m, n):
        self.numrows = m
        self.numcolumns = n
        self.matrix = np.zeros( (m,n), dtype=fractions.Fraction) 
        self.negmatrix = np.zeros( (m,n), dtype=fractions.Fraction) 
        self.max = 0
        self.min = 0
        self.negshift = 0

    # create matrix from any numerical matrix
    def __init__(self, A):
        AA = np.array(A)
        m,n = AA.shape
        self.numrows = m
        self.numcolumns = n
        self.matrix = np.zeros( (m,n), dtype=fractions.Fraction) 
        for i in range(m):
            for j in range(n):
                self.matrix[i][j] = utils.tofraction(AA[i][j])
        self.fullmaxmin() 

    def __str__(self):
        buf = columnprint.columnprint(self.numcolumns)
        for i in range(self.numrows):
            for j in range(self.numcolumns):
                buf.sprint(str(self.matrix[i][j]))
        out = str(buf)
        out += "\n# max= " + str(self.max) + ", min= " + str(self.min)
        out += ", negshift= " + str(self.negshift)
        return out

    def updatemaxmin(self, fromrow, fromcol): 
        m=self.numrows
        n=self.numcolumns
        for i in range(fromrow, m):
            for j in range(fromcol, n):
                elt = self.matrix[i][j]
                self.max = max(self.max, elt)
                self.min = min(self.min, elt)
        self.negshift = int(self.max)+1
        self.negmatrix = np.full((m,n),self.negshift,dtype=int)-self.matrix

    def fullmaxmin(self):
        self.max = self.matrix[0][0]
        self.min = self.matrix[0][0]
        self.updatemaxmin(0,0)

    # add full row, row must be of size n
    def addrow(self, row): 
        self.matrix = np.vstack([self.matrix, row])
        self.numrows += 1
        self.updatemaxmin(self.numrows-1,0)

    # add full column, col must be of size m
    def addcolumn(self, col):
        self.matrix = np.column_stack([self.matrix, col])
        self.numcolumns += 1
        self.updatemaxmin(0,self.numcolumns-1)

class bimatrix:
    # create A,B given m,n 
    def __init__(self, m, n):
        self.A = payoffmatrix(m,n)
        self.B = payoffmatrix(m,n)

    # create A,B from file
    def __init__(self, filename):
        lines = utils.stripcomments(filename)
        # flatten into words
        words = utils.towords(lines)
        m = int(words[0])
        n = int(words[1])
        needfracs =  2*m*n 
        if len(words) != needfracs + 2:
            print("in bimatrix file "+repr(filename)+":")
            print("m=",n,", n=",n,", need",
               needfracs,"payoffs, got", len(words)-2)
            exit(1)
        k = 2
        C = utils.tomatrix(m, n, words, k) 
        self.A = payoffmatrix(C)
        k+= m*n
        C = utils.tomatrix(m, n, words, k) 
        self.B = payoffmatrix(C)

    def __str__(self):
        out = "# m,n= \n" + str(self.A.numrows)
        out += " " + str(self.A.numcolumns)
        out += "\n# A= \n" + str(self.A)
        out += "\n# B= \n" + str(self.B)
        return out

    def createLCP(self):
        m = self.A.numrows
        n = self.A.numcolumns
        lcpdim = m+n+2
        lcp = lemke.lcp(lcpdim)
        lcp.q[lcpdim-2] = -1
        lcp.q[lcpdim-1] = -1
        for i in range(m):
            lcp.M[lcpdim-2][i] = 1
            lcp.M[i][lcpdim-2] = -1
        for j in range(m,m+n):
            lcp.M[lcpdim-1][j] = 1
            lcp.M[j][lcpdim-1] = -1
        for i in range(m):
            for j in range(n):
                lcp.M[i][j+m] = self.A.negmatrix[i][j]
        for j in range(n):
            for i in range(m):
                lcp.M[j+m][i] = self.B.negmatrix[i][j]
        # d for now
        for i in range(lcpdim):
            lcp.d[i]=1
        return lcp


    def runLH(self, droppedlabel):
        lcp = self.createLCP()
        lcp.d[droppedlabel-1] = 0  # subsidize this label
        tabl = lemke.tableau(lcp)
        # tabl.runlemke(verbose=True, lexstats=True, z0=gz0)
        tabl.runlemke(silent=True)
        return tuple(getequil(tabl))
        
    def LH(self, LHstring):
        if LHstring == "":
            return
        m = self.A.numrows
        n = self.A.numcolumns
        lhset = {} # dict of equilibria and list by which label found
        labels = rangesplit(LHstring, m+n)
        for k in labels:
            eq = self.runLH(k)
            if eq in lhset:
                lhset[eq].append(k)
            else:
                print ("label",k,"found eq", str_eq(eq,m,n))
                lhset[eq] = [k] 
        for eq in lhset:
            print (str_eq(eq,m,n),"found by labels", str(lhset[eq]))
        return lhset

    def runtrace(self, xprior, yprior):
        lcp = self.createLCP()
        Ay = self.A.negmatrix @ yprior
        xB = xprior @ self.B.negmatrix 
        lcp.d = np.hstack((Ay,xB,[1,1]))
        tabl = lemke.tableau(lcp)
        tabl.runlemke(silent=True)
        return tuple(getequil(tabl))

    def tracing(self, trace):
        if trace < 0:
            return
        m = self.A.numrows
        n = self.A.numcolumns
        trset = {} # dict of equilibria, how often found
        if trace == 0:
            xprior = uniform(m)
            yprior = uniform(n)
            eq = self.runtrace(xprior, yprior)
            trset[eq]=1
            trace = 1 # for percentage
        else:
            for k in range(trace):
                if seed >=0:
                    random.seed(10*trace*seed+k)
                x = randomstart.randInSimplex(m)
                xprior = randomstart.roundArray(x, accuracy)
                y = randomstart.randInSimplex(n)
                yprior = randomstart.roundArray(y, accuracy)
                # print (f"{k=} {xprior=} {yprior=}")
                eq = self.runtrace(xprior, yprior)
                if eq in trset:
                    trset[eq] += 1
                else:
#                     print ("found eq", str_eq(eq,m,n), "index",
#                         self.eqindex(eq,m,n))
                    trset[eq] = 1 
        equilibrium = None
        times_found = 0
        for eq in trset:
            if trset[eq] > times_found:
                equilibrium = eq
#             print (trset[eq],"times found ",str_eq(eq,m,n))
#         print(trace,"total priors,",len(trset),"equilibria found")
        return str_eq(equilibrium, m,n)

    def eqindex(self,eq,m,n):
        rowset,colset = supports(eq,m,n)
        k,l = len(rowset),len(colset)
        if k!=l:
            return 0
        A1 = submatrix(self.A.negmatrix, rowset, colset)
        DA = np.linalg.det(A1)
        B1 = submatrix(self.B.negmatrix, rowset, colset)
        DB = np.linalg.det(B1)
        sign = 2*(k%2) - 1 # -1 if even, 1 if odd
        if DA*DB == 0:
            return 0
        if DA*DB > 0:
            return sign
        return -sign 
        
def uniform(n):
    return np.array([ fractions.Fraction(1,n) for j in range(n)])

def getequil(tabl):
    tabl.createsol()
    return tabl.solution[1:tabl.n-1]

def str_eq(eq,m,n):
    x = "("+",".join([str(x) for x in eq[0:m]])+")"
    y = "("+",".join([str(x) for x in eq[m:m+n]])+")"
    rowset,colset = supports(eq,m,n)
    return x+","+y+"\n    supports: "+str(rowset)+str(colset)

def supports(eq,m,n):
    rowset = [i for i in range(m) if eq[i]!= 0]
    colset = [j for j in range(n) if eq[m+j]!= 0]
    return rowset,colset

def submatrix(A,rowset,colset):
    k,l = len(rowset),len(colset)
    B = np.zeros((k,l))
    for i in range (k):
        for j in range(l):
            B[i][j] = A[rowset[i]][colset[j]]
    return B

if __name__ == "__main__":
    processArguments()
    printglobals()

    G = bimatrix(gamefilename)
    print(G)
    eqset = G.LH(LHstring)
    eqset = G.tracing(trace)
