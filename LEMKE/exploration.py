import fractions
import copy
import sys
from polymatrix import polymatrix, Equilibrium
from utils import get_lcm
from plot import visualize_stage_flow_with_attraction
from lemke import RayTerminationError

class Logger(object):
    def __init__(self, filename):
        # "w" mode clears the file before writing
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Required for Python 3 compatibility
        pass

max_attempts = 100


def get_inverse_basis(tabl):
    """
    Extracts the basis inverse B^-1 from the integer tableau.
    Values are B_inv = Tableau_Col / Determinant.
    """
    n = tabl.n
    # inv_B_int will be B_inv * determinant
    inv_B_int = [[0 for _ in range(n)] for _ in range(n)]
    
    for j in range(n):
        # In lemke.py, W variables start at index n+1
        var_idx = n + 1 + j
        pos = tabl.bascobas[var_idx]
        
        if pos < n: # W_j is basic at row 'pos'
            inv_B_int[pos][j] = tabl.determinant
        else: # W_j is non-basic at tableau column 'pos-n'
            col_idx = pos - n
            for i in range(n):
                inv_B_int[i][j] = tabl.A[i][col_idx]
    return inv_B_int

                

def calculate_d_vector(game, priors, epsilon_val=fractions.Fraction(1, 1000)):
    """
    Calculates the d-vector using pure Python loops.
    Ensures exact Fraction math and avoids NumPy overflow.
    """
    d_parts = []
    for i in range(game.players):

        aij_xj = [fractions.Fraction(0)] * game.actions[i]
        for j in range(game.players):
            if i != j:

                matrix = game.A[i][j].negmatrix
                
                # Manual matrix-vector product: (Matrix A_ij) * (Strategy Y_j)
                for row_idx in range(len(matrix)):
                    for col_idx in range(len(priors[j])):
                        val = matrix[row_idx, col_idx]
                        p_val = priors[j][col_idx]
                        aij_xj[row_idx] += fractions.Fraction(val) * fractions.Fraction(p_val)
        d_parts.append(aij_xj)
        
    # Append the artificial 1s for the players (standard LCP construction)
    d_parts.append([fractions.Fraction(1)] * game.players)
    
    flat_d = []
    for sublist in d_parts:
        flat_d.extend(sublist)
        
    return flat_d

def jump_exploration(game, EQ1, EQ2, priorID=0):
    
    tableau = copy.deepcopy(EQ1.tabl) 
    n = tableau.n
    z0_col = tableau.bascobas[0] - n
    
    d_new = calculate_d_vector(game, EQ2.priors[priorID])
    sf_new = 1
    for val in d_new: 
        sf_new = get_lcm(sf_new, val.denominator)
        
    #  Inject into the z0 column
    inv_B_int = get_inverse_basis(tableau)
    for i in range(n):
        dot_product = -sum(inv_B_int[i][j] * (d_new[j] * sf_new) for j in range(n))
        tableau.A[i][z0_col] = int(dot_product)
    tableau.scalefactor[0] = sf_new
    tableau.pivotcount = 0
    tableau.lextested = [0]*(n+1)
    tableau.lexcomparisons = [0]*(n+1)

    # Handle Starting Direction
    enter = 0 
    leave, z0leave = tableau.lexminvar(enter)
    if tableau.bascobas[0] < n:
        print(f"LOG: Jump aborted. z0 already basic in Base {EQ1.ID}")
        return None
    
    if leave == -1: 
        return None # Genuine unbounded ray

    # Pivot Loop with a hard break to prevent infinite loops on Rays
    max_pivots = 500
    pcount = 0
    while True:
        try:
            tableau.testtablvars()
            tableau.pivot(leave, enter)
            pcount += 1
            if pcount > max_pivots: return None
            
            if z0leave: break
            enter = tableau.complement(leave)
            leave, z0leave = tableau.lexminvar(enter)
            if leave == -1: return None
        except: return None

    tableau.createsol()
    return tableau
     
def print_strategy_list(game_obj, eq_list):
    print("\n" + "="*120)
    print(f"{'ID':<4} | {'Stg':<3} | {'Prs':<3} | {'Ancestry [Base, Target, PriorIdx]':<15}| {'Strategy'} ")
    print("-" * 120)
    for eq in eq_list:
        eq_str = game_obj.format_eq_string(eq.eq)
        # Format ancestry triplets
        ancestry = str(eq.parent) if eq.parent else "Initial"
        if len(ancestry) > 12: ancestry = ancestry[:12] + "..."
            
        # Count of priors leading to this EQ
        p_count = len(eq.priors)
            
        print(f"{eq.ID:<4} | {eq.stage:<3} | {p_count:<3} | {ancestry:<15}  | {eq_str}")
    print("="*120 + "\n")


def print_relationship_table(log):
    print("\n" + "="*75)
    print(f"{'Base ID':<8} | {'Targ ID':<8} | {'Prior ID':<8} | {'Result':<8} | {'Notes'}")
    print("-" * 75)
    
    for p1, p2, p_idx, child, flag in log:
        print(f"{p1:<8} | {p2:<8} | {p_idx:<8} | {child:<8} | {flag}")
    
    total = len(log)
    returns = sum(1 for entry in log if "RETURN" in entry[4])
    new_found = sum(1 for entry in log if "NEW" in entry[4])
    rays = sum(1 for entry in log if "Ray" in entry[4])
    
    print("-" * 75)
    print(f"Total Jumps: {total} | New Found: {new_found} | Returns to S1: {returns}| Rays: {rays}")
    print("="*75 + "\n")
    
    
def test_return_to_Stage1(game, discovered_list):
    print("\n" + "="*60)
    print(f"{'Source (Stage 2)':<15} | {'Target Prior':<15} | {'Result'}")
    print("-" * 60)
    
    # Filter for Stage 2 equilibria
    stage_2 = [e for e in discovered_list if e.stage == 2]
    
    for child in stage_2:
        # Parent 2 (the one who provided the prior p_A) is child.parent[1] 
        p1_id = child.parent[0][0]
        p2_id = child.parent[0][1]
        parent_prior = child.parent[0][2]
        
        # Find the actual parent object for Parent 1 to get its priors
        parent2_obj = next(e for e in discovered_list if e.ID == p2_id)
        
        try:
            # Apply p_A (parent 1) on C (child)
            res_tabl = jump_exploration(game, child, parent2_obj, parent_prior)
            if res_tabl:
                new_vec = game.getequil(res_tabl)
                
                # Check if it returned to Parent 2
                parent1_obj = next(e for e in discovered_list if e.ID == p1_id)
                if all(abs(new_vec[k] - parent1_obj.eq[k]) < 1e-9 for k in range(len(new_vec))):
                    result = f"Returned to ID {p1_id}"
                else:
                    result = "Found NEW Stage 3!"
            else:
                result = "Ray Termination"
        except Exception:
            result = "Error"
            
        print(f"ID {child.ID:<13} | ID {p1_id:<13} | {result}")
    print("="*60 + "\n")

def explore_all(gamefile, trace = 100):
    game = polymatrix(gamefile)
    # Stage 1: Initial Tracing results (Priors already stored as lists here)
    all_discovered = game.tracing(trace) 
    relationship_log = []
    current_id_counter = len(all_discovered) + 1
    
    # We only use Stage 1 priors as "directions" for jumping
    stage_1_pool = [e for e in all_discovered if e.stage == 1]

    # BFS style exploration: i increments as all_discovered grows
    i = 0
    while i < len(all_discovered):
        base = all_discovered[i]
        
        for target in stage_1_pool:
            if base.ID == target.ID:
                continue

            for p_idx in range(len(target.priors)):
                # Redundancy Check: Don't jump back to the parent that produced this base
                # Triplet format: [BaseID, TargetID, PriorIdx(0-based)]
                if any(p[1] == target.ID and p[2] == p_idx for p in base.parent):
                    continue

                try:
                    
                    res_tabl = jump_exploration(game, base, target, p_idx)
                    if res_tabl:
                        raw_vec = game.getequil(res_tabl)
                        new_vec = tuple(fractions.Fraction(v).limit_denominator(1000) for v in raw_vec)
                        
                        # Match Check
                        match = next((e for e in all_discovered if 
                                     all(abs(new_vec[k] - e.eq[k]) < 1e-12 
                                         for k in range(len(new_vec)))), None)
                        
                        triplet = [base.ID, target.ID, p_idx]
                        flag = ""

                        if match:
                            child_id = match.ID
                            # Record the path even if the equilibrium was already found
                            if triplet not in match.parent:
                                match.parent.append(triplet)
                            
                            # Flag Returns to Stage 1
                            if match.stage == 1:
                                flag = "-> RETURN TO S1"
                                print("Returned to Stage 1:", base.ID, target.ID, p_idx, "led to ", match.ID)

                        else:
                            # NEW Discovery: Child stage is always Base + 1
                            child = Equilibrium(new_vec, target.priors[p_idx], res_tabl)
                            child.ID = current_id_counter
                            child.stage = base.stage + 1
                            child.parent.append(triplet)
                            
                            all_discovered.append(child)
                            child_id = child.ID
                            current_id_counter += 1
                            flag = f"NEW (S{child.stage})"
                    else:
                        child_id = "None"    
                        flag = "Ray"
                        
                except RayTerminationError:
                    child_id = "None"
                    flag = "Hit a Ray"
                
                relationship_log.append((base.ID, target.ID, p_idx, child_id, flag))
        i += 1
        if(len(all_discovered)>100): break
    
    # Final Output to file/terminal
    print_strategy_list(game, all_discovered)
    print_relationship_table(relationship_log)
    test_return_to_Stage1(game, all_discovered)
    #visualize_equilibrium_map(all_discovered, relationship_log, "game_discovery_map.png")
    visualize_stage_flow_with_attraction(all_discovered, relationship_log)





if __name__ == "__main__":
    #sys.stdout = Logger("output.txt")
    explore_all("PMGames/poly.txt", 100)