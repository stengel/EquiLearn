import utils
import lemke
import fractions
import random # random.seed
from bimatrix import payoffmatrix
from randomstart import randInSimplex, roundArray

# file format: 
# m1 m2 m3 ..... m_n denoting n player i's number of possible actions
# m1*m2 entries for A, separated by blanks / newlines
# m2*m1 entries of B, separated by blanks / newlines
# .... and so on
# blank lines or lines starting with "#" are ignored
seed = -1
trace = 100 # negative: no tracing
accuracy = 1

class Equilibrium():
    def __init__(self, eqm, p, tab):
        self.ID = 0
        self.eq = eqm
        self.priors = [p] if p is not None else []
        self.tabl = tab
        self.parent = []
        self.stage = 1
        
# A -> lends its Prior
# B -> lends its tableau (run all priors that lead to non-B)
# stage 2:
# C is found -> we save the prior of A and B's B-inv (maintain stage in class)
# Test p_A on C, it should go back to B. 
# list[0] = origin,         self.id = 1,2,3.....        self.parents = number if its stage 2../ 0

class polymatrix():
    # create A given m vector
    def __init__(self, m):
        self.actions = m
        self.players = len(m)
        
        #mainmatrix = np.zeros((self.players, self.players), dtype=object) 
        mainmatrix = [[0 for _ in range(self.players)] for _ in range(self.players)]
        for i in range(self.players):
            for j in range(self.players):
                if i!=j:
                    #mainmatrix[i][j] = utils.matrix_tofraction(np.random.randint(0, 50, [m[i], m[j]]))
                    random_data = [[random.randint(0, 49) for _ in range(m[j])] for _ in range(m[i])]
                    mainmatrix[i][j] = utils.matrix_tofraction(random_data)
        self.A = mainmatrix

    def __init__(self, M, m): #f = file containing list of matrices 

        self.actions = m
        self.players = len(m)
        mainmatrix = [[0 for _ in range(self.players)] for _ in range(self.players)]
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
        mainmatrix = [[0 for _ in range(self.players)] for _ in range(self.players)]
        k = 0
        for i in range(self.players):
            for j in range(self.players):
                if i!=j:
                    mainmatrix[i][j] = payoffmatrix(M[k])
                    k+=1
        self.A = mainmatrix
        self.writeToFile(f)
    
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
        lcpdim = sum(self.actions) + self.players
        lcp = lemke.lcp(lcpdim)
        
        # Set q = -1 only for the probability constraint rows (the last N rows)
        for i in range(self.players):
            lcp.q[sum(self.actions) + i] = -1

        for i in range(self.players):
            # Rows corresponding to Player i's actions
            start_row = sum(self.actions[:i])
            currentrows = range(start_row, start_row + self.actions[i])
            
            # Column for Player i's game value variable (v_i)
            v_col = sum(self.actions) + i
            
            for idx_k, k in enumerate(currentrows):
                lcp.M[k][v_col] = -1  # A_i * x_j - v_i <= 0
                
                for j in range(self.players):
                    if i == j: continue
                    # Columns for Player j's actions
                    start_col = sum(self.actions[:j])
                    currentcols = range(start_col, start_col + self.actions[j])
                    
                    for idx_l, l in enumerate(currentcols):
                        if self.A[i][j] != 0:
                            # Payoffs are placed in the strategy intersections
                            lcp.M[k][l] = self.A[i][j].negmatrix[idx_k][idx_l]
            
            # Probability constraint: sum(x_i) = 1 (This row MUST only contain 1s for player i)
            v_row = sum(self.actions) + i
            for k in currentrows:
                lcp.M[v_row][k] = 1

        # Standard covering vector
        for i in range(lcpdim):
            lcp.d[i] = 1
    
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
            aij_xj = [fractions.Fraction(0)] * self.actions[i]
            for j in range(self.players):
                if i != j:
                    # Manual matrix-vector product to avoid NumPy's @ operator
                    matrix = self.A[i][j].negmatrix
                    for row_idx in range(len(matrix)):
                        # Use Fractions throughout to maintain exact precision
                        sum_val = fractions.Fraction(0)
                        for col_idx in range(len(priors[j])):
                            val = matrix[row_idx, col_idx]
                            p_val = priors[j][col_idx]
                            sum_val += fractions.Fraction(val) * fractions.Fraction(p_val)
                        aij_xj[row_idx] += sum_val
               
            AX.append(aij_xj)
        
        # Add the covering vector constant for players
        AX.append([fractions.Fraction(1)] * self.players)
        
        flat_d = []
        for sublist in AX:
            flat_d.extend(sublist)
            
        lcp.d = flat_d
        tabl = lemke.tableau(lcp)
        tabl.runlemke(verbose=False, lexstats=True, z0=False, silent=True)
        return tuple(self.getequil(tabl)), tabl
    

    def tracing(self, trace):
        deq = {}
        eq_obj = []
        if trace < 0:
            return
        elif trace == 0:
            priors = []
            for p in range(self.players):
                priors.append(list(self.uniform(self.actions[p])))
            result, tabl = self.runtrace(priors)
            eq = utils.fractions_np_to_int(result)
            deq[eq]=1
            eq_obj.append(Equilibrium(eq,priors,tabl))
            trace = 1 
        else:
            for k in range(trace):
                if seed >=0:
                    random.seed(10*trace*seed+k)
                priors = self.getpriors(accuracy)
                result, tabl = self.runtrace(priors)
                eq = utils.fractions_np_to_int(result)
                
                if eq in deq:
                    deq[eq] += 1
                    for existing in eq_obj:
                        if all(abs(eq[i] - existing.eq[i]) < 1e-12 for i in range(len(eq))):
                            existing.priors.append(priors)
                            break
                    
                else:
                    deq[eq] = 1
                    new_e = Equilibrium(eq, priors, tabl)
                    new_e.ID = len(eq_obj) + 1
                    new_e.stage = 1
                    eq_obj.append(new_e)
                    
        self.print_statistics(deq)
        return eq_obj
    
    def uniform(n):
        return [fractions.Fraction(1,n) for j in range(n)]
   
        
    def getequil(self,tabl):
        tabl.createsol() # probably redundant, as runlemke already calls createsol.
        return tabl.solution[1:sum(self.actions)+1]
    
    def print_statistics(self, deq):
        for k in deq:
            st = self.format_eq_string(k)
            print("--------")
            print(st)
            print("found ", deq[k], " times")
            print("supports: ",self.find_supports(k))

    def format_eq_string(self, eq_vector):
        """Helper to format equilibrium vector as fractions."""
        st = ""
        start = 0
        total_probs = len(eq_vector)
        for idx in self.actions:
            st += "("
            for j in range(start, start + idx):
                if j < total_probs:
                    val = fractions.Fraction(eq_vector[j]).limit_denominator()
                    if abs(val) < 1e-12:
                        st += "0 "
                    else:
                        st += f"{val} "
            st = st.strip() + ") "
            start += idx
        return st.strip()
        
    def find_supports(self, eq):
        supports = []
        idx = 0
        for i in range(self.players):
                supp = [j for j in range(self.actions[i]) if eq[idx+j]!= 0]
                supports.append(supp)
                idx += self.actions[i]
        return supports


        
#M = [[[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]]


if __name__ == "__main__":
    x = polymatrix("PMGames/poly.txt")
    x.tracing(trace)
    
    

   














