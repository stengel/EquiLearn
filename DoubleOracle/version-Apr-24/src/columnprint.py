# automatic pretty printing in columns 
import random

class columnprint:
    
# create buffer with c columns, at least 1
    def __init__(self, c): # void colset(int c);
        if c < 1 :
            print ("need positive number of columns, not", c)
            raise Exception
        self.buf = []
        self.ncols = c
        self.colwidth = [0]*c
        self.widthsign = [1]*c
        self.currlines = 0
        self.currcol = 0
        self.line = []

# create string, intermittent lines (not last line) separated by \n
    def __str__(self) : # colout(void)
       out = ""
       for l in self.buf: 
           out += self.prline(l) + "\n"
       if self.line == [] :
           out = out[:-1]
       else: 
           out += self.prline(self.line) 
       return out

# (private) string from line, which is a list of strings
    def prline(self, line): # prline(char *s)
        out = ""
        i = 0
        for word in line:
            if self.widthsign[i] == 1:
                word = word.rjust(self.colwidth[i])
            else:
                word = word.ljust(self.colwidth[i])
            out += word
            i += 1
            if i < self.ncols: 
                out += " " ####### single spacing here
        return out

# print integer i into the current column 
    def iprint(self, i): # colipr(i)
        self.sprint(str(i))

# make column c in  0..ncols-1  left-adjusted  
    def makeLeft(self, c): # colleft(int c)
        self.widthsign[c] = -1

# terminate current line early.  blank line if in column 0  
    def newline(self):   # colnl(void);
        for j in range(self.ncols - self.currcol):
            self.sprint("")

# store string  s  into the current column, updating column width
    def sprint(self,s): # colpr(const char *s)
        w = len(s)
        if self.colwidth[self.currcol] < w:
            self.colwidth[self.currcol] = w
        self.line.append(s)
        self.currcol += 1
        if self.currcol == self.ncols :
            self.buf.append(self.line)
            self.line = []
            self.currcol = 0

if __name__ == "__main__":       
    print ("example with 3 columns, first left-justified")
    print ("============================================")
    M = columnprint(3)
    M.makeLeft(0)
    M.sprint("z0")
    M.sprint("z1")
    M.sprint("z2")
    for i in [1,200,3,20000,0,-1,88,9] :
        M.iprint(i)
    print(M)
    print ("example with 2 columns")
    print ("======================")
    M = columnprint(2)
    M.sprint("z0")
    M.sprint("z1")
    for i in [1,200,3,20000,0,-1,88,9] :
        M.iprint(i)
    print(M)
