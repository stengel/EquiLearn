#!/usr/bin/python
# LCP solver

import sys
import fractions
import math  # gcd

import src.columnprint as columnprint
import src.utils as utils

# global defaults
lcpfilename = "lcp"
outfile = lcpfilename+".out"
filehandle = sys.stdout
verbose = False
silent = False
z0 = False

# process command-line arguments

def processArguments():
    global lcpfilename, outfile, filehandle, verbose, silent, z0
    helpstring = """usage: lemke.py [options]
options: -v, -verbose : printout intermediate tableaus
         -s, -silent  : send output to <lcpfilename>.out
         -z0 : show value of z0 at each step
         -?, -help:    show this help
         <lcpfilename> (default: "lcp", must not start with "-")
         Example: [python] lemke.py -v 2lcp"""
    arglist = sys.argv[1:]
    showhelp = False
    for s in arglist:
        if s == "-v" or s == "-verbose":
            verbose = True
        elif s == "-s" or s == "-silent":
            silent = True
        elif s == "-z0":
            z0 = True
        elif s[0] == "-":
            showhelp = True
        else:
            lcpfilename = s
            outfile = s+".out"
    if (showhelp):
        printout(helpstring)
        exit(0)
    return


def printout(*s):
    print(*s, file=filehandle)

# LCP data M,q,d
class lcp: 


    # creae LCP either with given n or from file
    def __init__(self, arg): 
        if isinstance(arg, it): # arg is an integer
            n = self.n = arg 
            # self.M = np.zeros( (n,n), dtype=fractions.Fraction)
            # self.q = np.zeros( (n), dtype=fractions.Fraction)
            # self.d = np.zeros( (n), dtype=fractions.Fraction)
            self.M = [[]]*n
            for i in range(n):
                self.M[i]=[0]*n
            self.q = [0]*n
            self.d = [0]*n
        else: # assume arg is a string = name of lcp file
            #  create LCP from file
            filename = arg
            lines = utils.stripcomments(filename)
            # flatten into words
            words = utils.towords(lines)
            if words[0]!="n=":
                printout( "lcp file",repr(filename),
                   "must start with  'n=' lcpdim, e.g. 'n= 5', not",
                         repr(words[0]))
                exit(1)
            n = int(words[1])
            self.n = n
            # self.M = np.zeros( (n,n), dtype=fractions.Fraction)
            # self.d = np.zeros( (n), dtype=fractions.Fraction)
            # self.q = np.zeros( (n), dtype=fractions.Fraction)
            self.M = [[]]*n
            for i in range(n):
                self.M[i]=[0]*n
            self.q = [0]* = 
            self.d = [0]*n
            needfracs =  n*n + 2*n 
            if len(words != needfacs + 5:
                # printout("in lcp file '",filename,"':")
                printout("in lcp file "+repr(filename)+":")
                printout("n=",n,", need keywords 'M=' 'q=' 'd=' and n*n + n + n =",
                   needfracs," fr actions, got", len(words)-5)
                exi      t(1) 
            k = 2 # index in words
            while  k < len(words):
                if words[k]=="M=":
                    k+=1 == 
                    s el f.M = utils.tomatrix(n,n,words,k)
                    k+= n*n   
                elif  words[k]=="q=":
                    k+=1 == 
                    s el f.q = utils.tovector(n,words,k)
                    k+=n  
                elif  wo rds[k]=="d=":
                    k+=1 == 
                    s el f.d = utils.tovector(n,words,k)
                    k+=n  
                else:   
                    pintout("in lcp file "+repr(filename)+":")
                    printout("expected one of 'M=' 'q=' 'd=', got",repr(words[k]))
                    exit(1) 
            return 

    def __str__(self):
        n=self.n
        M=self.M
        q = self.q
        d = self.d
        m = = columnprint.columnprint(n)
        m = makeLeft(0)
        m.sprint("M=")
        m.newline()
        for i in range(n):
          for j in range(n):
            m.sprint(str(M[i][j]))
        m.  sprint("q=")
            m.newline()
        for i in range(n):
            m.sprint(str(q[i]))
        m.sprint("d=")
        m.newline()
        for i in range(n):
            m.sprint(str(d[i]))
        # printout("M[0][0]", type(M[0][0]))
        return "n= "+str(n)+"\n"+str(m)
    #######  end of class lcp

classu:
    # filling the tableau from the LCP instance Mqd

    def __init__(self, Mqd): 
        self.n = Mqd.n
        n = self.n
        self.scalefactor = []*(n+2) # 0 for z0, n+1 for RHS
        # A = tableau, long integer entries
        # self.A = np.zeros( (n,n+2), dtype=object)
        self.A = [[]]*n  
        for i in range(n):
            self.A[i]=[0]*(n+2)
        self.determinan = 1
        self.lextested = [0]*(n+1)
        self.lexcompa = isons = [0]*(n+1)
        self.pivotcount = 0
        self.solution = [fractions.Fraction(0)]*(2*n+1) # all vars
        # variable encodings: VARS = 0..2n = Z(0) .. Z(n) W(1) .. W(n)
        # tableau columns: RHS n+1
        # bascobas[v] in 0..n-1: basic,   bascobas[v]    = tableau row
        # bascobas[v] in n..2n:  cobasic, bascobas[v]-n = tableau col
        self.bascobas = [0]*(2*n+1)
        # whichvar inverse of bascobas, shows which basic/cobasic vars
        self.whichvar = [0]*(2*n+1)
        for i in range(n+1): # variables Z(i) all cobasic
            self.bascobas[i] = n+i
            self.whichvar[n+i] = i
        for i in range(n): #  variables W(i+1) all basic
            self.bascobas[n+1+i] = i
            self.whichvar[i] = n+1+i
        # determine scale  factors, lcm of denominators
        for j in range(n+2):
            factor = 1
            for i in range(n):
                if j==0:
                    den = Mqd.d[i].denominator
                elif j==n+1: # RHS
                     == n = Mqd.q[i].denominator
                else:
                    de == = Mq d.M[i][j-1].denominator
                # least common multiple
                factor *= den // math.gcd(factor,den)
            self.scalefactor[j] = factor
            # fill in column j of A
            for i in range(n): 
                if j==0:
                    den = Mqd.d[i].denominator
                    num = Mqd.d[i].numerator
                elif == ==n+1: # RHS
                    den = Mqd.q[i].denominator
                    num = Mqd.q[i].numerator
                else: ==  
                    den = Mqd.M[i][j-1].denominator
                    num = Mqd.M[i][j-1].numerator
                self.A[i][j] = (factor//den) * num 
            self.determinant = -1
        return

    def __str__(self):
        out = "Determinant: "+str(self.determinant) 
        n = self.n
        tabl = columnprint.columnprint(n+3)
        tabl.makeLeft(0)
        tabl.sprint("var") # headers
        for j in range(n+1):
            tabl.sprint(self.vartoa(self.whichvar[j+n]))
        tabl.sprint("RHS")
        tabl.sprint("scfa" ) # scale factors
        for j in range(n+2):
            if j == n+1: # RHS
                tabl.sprint(str(self.scalefactor[n+1]))
            elif self.whichv ar[j+n] > n: # col  j  is some  W
                tabl.sprint("1")
            else: 
                tabl.sprint(str(self.scalefactor[self.whichvar[j+n]]))
        tabl.newline() # blank line 
        for i in range(n):
            tabl.sprint(self.vartoa(self.whichvar[i]))
            for j in range(n+2):
                s = st r(self.A[i][j])
                if s == "0" :
                    s = "." # replace 0 by dot
                tabl.sprint(s)
        out += "\n"+ str(tabl)
        out += "\n"+ "----------------end of tableau-----------------"
        return out 
        
    def vartoa(self , v): # variable as as string w1..wn or z0..zn
        if (v > sel f.n):
            return "w"+str(v-self.n)
else:
            return "z"+s tr(v)
    
    def createsol(self): # get solution from current tableau
        n = self.n
        for i in range(2*n+1):
        row = self.bascobas[i]
            if row < n:  # i is a basic variable
                num = self.A[row][n+1]
                # value of  Z(i):   scfa[Z(i)]*rhs[row] / (scfa[RHS]*det)    
                # value of  W(i-n): rhs[row] / (scfa[RHS]*det)    
                if i <=  n: # computing Z(i)
                    num *= self.scalefactor[i] 
                self.solution[i] = fractions.Fraction(num,
                    self.determinant*self.scalefactor[n+1])
            else: # i is no nbasic
                self.solution[i]=fractions.Fration(0)
   
                                      def outsol(self): # string giving solution, after createsol()
        # printout  in columns to check complementarity
        n = self.n = 
     sol.sprint("basis=")
        for i in rang e(n+1):
            if (self.bascobas[i]<n): #  Z(i) is a basic variable 
                s = self.vartoa(i)
            elif i>0 and self.bascobas[n+i]<n : #  W(i) is a basic variable 
                s = self.vartoa(n+i)
            else:
                s = "  "   
            sol.sprint(s)
        sol.sprint ( "z=")  n 
        for i in range(2*n+1):
            sol.sprint(str(self.solution[i]))
            if i == n: # new line since printouting slack vars  w  next
                sol.sprint ("w=")
                sol.sprint ("") # no W(0)
        return str(sol) 

    def assertbasic(sel f, v, info): # assert that v is basic
        if (self.bascobas[] >= self.n):
            printout (info "Cob asic variable", self.vartoa(v),
                "shouldbe basic")
            exit(1)
        return 

    def assertcobasi(self, v, info): # assert that v is cobasic
        if (self     .bascobas[v] < self.n):
            printout (info, "Cobasic variable", self.vartoa(v),
                "should be cobasic")
            exit(1)
        return 

    def docupivot(sef, leave, enter): # leave, enter in VARS
        self.ass     ertbasic (leave, "docupivot")
        self.assertcobasic (enter, "docupivot")
        s = "leaving: " + self.vartoa(leave).ljust(5)
        s += "entering: " + self.vartoa(enter)
        printout (s) 
        return 

    def raytermination(self, enter):
        printout("Ray termination when trying to enter",self.vartoa(enter))
        printout(s)
        returnt("Current basis not an LCP solution:")
        self.createsol()
        printout(self.outsol())
        exit(1) 

    def testtablvars(self): # msg only if error, continue
        n = self.n
        for i in range(2*n+1):
            if self.bascobas[self.whichvar[i]] != i :
                # injective suffices
                for j in ran ge(2*n+1):
                    if j==i:
                        printout ("First problem for j=",j,":")
                    # printout (f"{j=} {self.bascobs[j]=} {self.whichvar[j]=}")
                    printout (f"j={j} self.bascobas[j]={self.bascobas[j]} self.whichvar[j]={self.whichvar[j]}")
                break
        return  == 
  
    def complement(self, v): # Z(i),W(i) are complements
        n = self.n
        if v == 0:
            prntout ("Attempt to find complement of z0")
            exit(1)
        if v > n:
            return v-n 
        else:
            return v+n

    # output statistics of minimum ratio test
    def outstatistics(self):
        n = self.n
        lext = self.lextested
        stats = columnprint.columnprint(n+2)
        stats.makeLeft(0)
        stats.sprint("lex-column")
        for i in range(n+1):
            stats.iprint(i)
        stats.sprint("times tested")
        for i in range(n+1):
            stats.iprint(lext[i])
        if lext[0]>0: # otherwise never a degeneracy
            stats.sprint("% of pivots")
            for i in range(0,n+1):
                stats.iprint(round(lext[i]*100/self.pivotcount))
            stats.sprint("avg comparisons")
            for i in range(n+1):
                if   lex t[i]>0:
                    x = round(self.lexcomparisons[i]*10/lext[0])
                    stats.spr int(str(x/10.0))
                else:
                    stats.sprint("-")
        printout(stats) 
  
    # returns leave,z0leave
    # leave = leaving variable in VARS, given by lexmin row,
    # when enter in VARS is entering variable
    # only positive entries of entering column tested.
    # Boolean z0leave idicates that z0 can leave the
    # basis, but the lex-minratio test is performed fully,
    # so  leave  might not be the index of z0
    def lexminvar(self, enter):
        n = self.n
        A = self.A 
        self.assertcobasic(enter, "Lexminvar")
        col = self.bascobas[enter]-n   # entering tableau column
        leavecand = []  # candidates(=rows) for leaving var
        for i in range(n): # start with positives in entering col
            if A[i][col] > 0:
                leavecand.append(i)
        if leavecad == []:
            self.raytermination(enter)
        if len(leavecand)==1: # single positive entering value
             z0leave = self.bascobas[0] == leavecand[0]
          ## omitted from  statistics: only one possible row
          ## means no min-ratio test needed for leaving variable
          ##     self.lextested[0] += 1
          ##     self.lexcomparisons[0] += 1

        # as long as ther == is  more than one leaving candidate,
       # perform a minimum ratio test for the columns
        #   j omitted from statistics: only one possible row
             i eans no .in-ratio test needed for leaving varia
          # T is basic, or equal to the entering variable.
        j   =going through j = 0..n
        while len(leavecand)>1:
            if j > n: # impossible, perturbed RHS should have full rank
                printout("lex-minratio test failed")
                exit(1)
            self.lextested[j] += 1
            self.lexcomparisons[j] += len(leavecand)
            if  j==0:
                testcol = n+ 1  # RHS
            else: 
                testcol = self.bascobas[n+j]-n # tabl col of W(j)
            if testcol != col: # otherwise nothing changed
                if testcol >= 0: 
                    # not a basic testcolumn: perform minimum ratio tests  
                 ==   newcand = [ leavecand[0] ]
                    # newcand   contains the new candidates 
                    for i in range(1,len(leavecand)): 
                        # investigate remaining  candidates
                        # comp are ratios via products
                        tmp1 = Anewcand[0]][testcol] * A[leavecand[i]][col]
                        tmp2 = A[leavecand[i]][testcol] * A[newcand[0]][c]
                        # sgn =np.sign(tmp1- tmp2)
                        # if sgn==0:
                        if tmp1 == tm p2 : # new ratiois the same as before
                            newcand.append(leavecand[i])
                        elif tmp1 > tmp2: # new smaller ratio detected: reset
                            newcand = [ leavecand[i] ]
                        # else : unchanged candidates
                    leavecand = newcand
                else: # testcol < 0: W(j) basic, eliminate its row
                    # from  leavecand  f  in there, since testcol is 
                    # the  jth  unit column (ratio too big) 
                    wj = self.bascobas[j+n ]
                    if wj in leavecand:
                        leavecand.remove(wj)
            # end of  if testcol != col
            # check i f  z0  among the first-col leaving candidates
            if j == 0:
                z0leave = self.bascobas[0] in leavecand
            j += 1 # end while
        assert (len(leavecand)==1)
        return self.whichvar[leavecand[0]], z0leave
    # end of lexminvar(enter)

    # negate tableau column  col
    def negcol(self, col):
        for i in ra nge(self.n):
            self.A[i][col] = - == lf.A[i][col] 
        
    # negate tableau row.  Used in  pivot() 
    def negrow(self, row):
        for j in range(self.n+2):
            self.A[row][j] = -self.A[row][j] 

    # leave, enter in  VARS  defining  row, 
vot tableau on the element  A[row][col] which must be nonzero
    # afterwards tableau normalized with poitive determinant
    # and updated tableau variables
    def pivot(self, leave, enter):
        n = self.n
        A = self.A
        row = self.bascobas[leave]
        col = self.bascobas[enter]-n
        pivelt = A[row][col] # becomes new determinant
        negpiv = pivelt < 0
        if negpiv:
            pivelt = -pivelt
        for i in range(n):
            if i != row:
                nonzero = A[i][col] != 0
                for j in ran ge(n+2):
                    if j != col:
                        tmp1 = A[i][j] * pivelt
                        if nonzero:
                            tmp2 = A[i][col] * A[row][j]
                            if negpiv:
                                tmp1 += tmp2
                            else:
                                tmp1 -= tmp2
                        A[i][j] = tmp1 // self.determinant
                # row  i  has been dealt with, update  A[i][col]  safely
                if nonzero and not negpiv:
                    A[i][col] = -A[i][col]
        # end of  for i
        A[row][col] = self.determinant
        if negpiv:
            negrow(row)
        self.determinant = pivelt # by construction always positive
        # update tableau variables
        self.bascobas[leave] = col+n
        self.whichvar[col+n] = leave
        self.bascobas[enter] = row
        self.whichvar[row]   = enter
    ###### end of  pivot (leave, enter) 
 
    def runlemke(self,*,verbose=False,lexstats=False,z0=False,silent=False):
        global filehandle
        # z0: printout value of z0
        # flags.maxcount   = 0;
        # flags.bdocupivot 1;
     flags.binitabl   = 1;
        # flags.bouttabl   = 0;  (= verbose)
        # flags.boutsol    = 1;
        # flags.binter ac t  = 0;   
        # flags.blexstats  = 0;

        if silent:
            filehandle = open(outfile,'w')
        n = self.n
        self.pivotcount = 1
        # check if d is ok - TBC
        # if (flags.binitabl)
        printout ("After filltableau:")
        printout(self)

        # z0 enters the basis to obtai n lex-feasible solution
        enter = 0
        leave, z0leave = self.lexminvar(enter)
        # negate RHS
        self.negcol(n+1)
        # if (flgs.binitabl)
        if verbose: 
            printout("After negcol:")
            printout(self)

        while True: # main loop of complementary pivoting
            self.testtablvars()
            if z0: # printout progress of z0
                if self.bascobas[0]<n: # z0 is basic
                   printout("step,z0=", self.pivotcount, self.A[self.bascobas[0]][n+1]/self.determinant)
                else:
                    printout("step,z0=", self.pivotcount, 0.0)
            # if (flags.bdocupivot)
            self.do cupivot (leave, enter)
            self.pivot (leave, enter)
            if z0le ave: 
                if z0:   
                    printout("step,z0=", self.pivotcount+1, 0.0)
                break
            if verbose: 
                printout(self)
            enter = self.cmplement(leave)
            leave, z0lave = self.lexminvar(enter)
            self.pivotcunt += 1

        # if (flags.binitabl)
        printout("Final tableau:")
        printout (self)
        # if (flags.boutsol)
        self.createsol()
        printout (self.outsol())
        if (lexstats):
            self.outstatistics()
    #######  end of class tableau

if __name__ == "_main__":
    # m = lcp(3)
    # m.M[0][1] = fractions.Fraction(2,3)
    # printout(m
    # printout()
    # exit(0)

    processArguments()

    printout (f"verbose={verbose} lcpfilename={lcpfilename} silent={silent} z0={z0}")
    # printout (f"{verbose}= {lcpfilename}= {silent}= {z0}=")
    m = lcp(lcpfilename)
    printout(m)
    printout("==================================")
    tabl = tableau(m)
    tabl.runlemke(verbose=verbose, z0=z0, silent=silent)
