import numpy as np
import fractions
import random
from polymatrix import polymatrix

max_attempts = 100

def get_inverse_basis(tabl):
    """
    Extracts the basis inverse B^-1 from the integer tableau.
    Values are B_inv = Tableau_Col / Determinant.
    """
    n = tabl.n
    inv_B = np.zeros((n, n), dtype=object) 
    
    for j in range(1, n + 1):
        var_idx = n + j # Slack variable W_j
        if var_idx >= len(tabl.bascobas): continue
            
        pos = tabl.bascobas[var_idx]
        
        if pos < n: # Basic
            inv_B[pos, j-1] = fractions.Fraction(1)
        else: # Non-basic
            col_idx = pos - n
            for i in range(n):
                if 0 <= col_idx < len(tabl.A[i]):
                    inv_B[i, j-1] = fractions.Fraction(tabl.A[i][col_idx], tabl.determinant)          
    return inv_B

def getManyEquilibria(game):
    eqa = game.tracing(100)
    return random.sample(eqa,2)

def getEquilibria(game):
    eqa = []
    attempts = 0
    
    # Run first attempt
    obj = game.tracing(1)[0]
    eqa.append(obj)
    first_eq_vec = obj.eq
    
    print("LOG: Found EQ1.")
    
    while len(eqa) < 2:
        attempts += 1
        eq_obj_list = game.tracing(1)
        
        if not eq_obj_list: continue
            
        eq_obj = eq_obj_list[0]
        
        # Simple check: are the strategy vectors different?
        if eq_obj.eq != first_eq_vec:
            print(f"LOG: Found distinct EQ2 after {attempts} attempts.")
            eqa.append(eq_obj)
        
        if attempts >= max_attempts:
            print("LOG: Max attempts reached searching for EQ2.")
            break
            
    return eqa

def calculate_d_vector(game, priors):
    d_parts = []
    for i in range(game.players):
        aij_xj = [0] * game.actions[i]
        for j in range(game.players):
            if i != j:
                try:
                    term = game.A[i][j].negmatrix @ priors[j]
                    aij_xj = [x+y for x,y in zip(aij_xj, term)]
                except: continue
        d_parts.append(aij_xj)
    d_parts.append([1] * game.players)
    return np.hstack(d_parts)

def jump_exploration(game, EQ1, EQ2):
    """
    Performs a 'Ray-Jump' by injecting a covering vector d2 into 
    the equilibrium basis_1 and tracing a new path.
    """
    import copy
    tableau = copy.deepcopy(EQ1.tabl) 
    
    n = tableau.n

    # 1. Degeneracy Catch
    pivots_to_clear = 0
    while tableau.bascobas[0] < n and pivots_to_clear < n:
        z0_row = tableau.bascobas[0]
        
        found_pivot = False
        for col in range(n):
            if tableau.A[z0_row][col] != 0:
                tableau.pivot(z0_row, col)
                found_pivot = True
                break
        
        if not found_pivot:
            print("LOG: FAILED to pivot z0 out. Basis appears singular.")
            return None
        pivots_to_clear += 1

    z0_pos = tableau.bascobas[0]
    z0_col = z0_pos - n
    
    if z0_col < 0:
        print(f"LOG: ERROR: z0 is still basic. Cannot jump.")
        return None

    # 2. Calculate new d (covering vector from EQ2's priors)
    d_new = calculate_d_vector(game, EQ2.priors)
    
    # 3. Transform d_new to the current coordinate system
    B_inv = get_inverse_basis(tableau)
    d_trans = B_inv @ d_new
    
    # 4. Inject into the tableau
    for i in range(n):
        exact_val = -d_trans[i] * fractions.Fraction(tableau.determinant)
        tableau.A[i][z0_col] = int(exact_val)

    # 5. Manual Pivot Loop
    # z0 enters the basis to obtain lex-feasible solution
    enter = 0
    leave, z0leave = tableau.lexminvar(enter)
    loopcount = 0
    while True: # main loop of complementary pivoting
        tableau.testtablvars()
        # printout progress of z0
        if tableau.bascobas[0]<n: # z0 is basic
            print("step,z0=", tableau.pivotcount, tableau.A[tableau.bascobas[0]][n+1]/tableau.determinant)
        else:
            print("step,z0=", tableau.pivotcount, 0.0)

        print(tableau.docupivot(leave, enter))
        tableau.pivot(leave, enter)
        if z0leave: 
            print("step,z0=", tableau.pivotcount+1, 0.0)
            break
        print(tableau)
        enter = tableau.complement(leave)
        leave, z0leave = tableau.lexminvar(enter)
        tableau.pivotcount += 1
        loopcount+=1
        if loopcount >= max_attempts:
            print("pivoted too many times")
            break
        
    print("Final tableau:")
    print(tableau)

    tableau.createsol()
    print(tableau.outsol())
    return tableau

def format_eq_string(game_obj, eq_vector):
    """Helper to format equilibrium vector as fractions."""
    st = ""
    start = 0
    total_probs = len(eq_vector)
    for idx in game_obj.actions:
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

def print_equilibria_table(game_obj, eq_list):
    print("\n" + "="*80)
    print(f"{'Equilibrium Strategy':<40} | {'Priors (Player 1)':<20} | {'Tableau Info'}")
    print("-" * 80)
    
    for i, eq_obj in enumerate(eq_list):
        eq_str = format_eq_string(game_obj, eq_obj.eq)
        if len(eq_str) > 38:
            eq_str = eq_str[:35] + "..."
            
        if eq_obj.priors and len(eq_obj.priors) > 0:
            p_sample = str([float(x) for x in eq_obj.priors[0]])
            if len(p_sample) > 18:
                 p_sample = p_sample[:15] + "..."
        else:
            p_sample = "N/A"
            
        det = str(eq_obj.tabl.determinant)
        if len(det) > 10: det = det[:7] + "..."
        tab_info = f"Det: {det}"
        
        print(f"{eq_str:<40} | {p_sample:<20} | {tab_info}")
    print("="*80 + "\n")

def explore(gamefile, many=False):
    game = polymatrix(gamefile)
    
    # 1. Find two equilibria
    if many:
        EQ_list = getManyEquilibria(game)
    else:
        EQ_list=getEquilibria(game)
    
    if len(EQ_list) < 2:
        print("LOG: Could not find 2 distinct equilibria to bridge. Exiting.")
        return

    print_equilibria_table(game, EQ_list)
    print(EQ_list[0].tabl)
    print(EQ_list[1].tabl)
    
    # # 2. Try Jump 1: EQ1 -> EQ2 direction
    # final_tableau1 = jump_exploration(game, EQ_list[0], EQ_list[1])
    # if final_tableau1:
    #     print("LOG: Jump 1 Result:")
    #     EQ3_vec = game.getequil(final_tableau1)
    #     print(format_eq_string(game, EQ3_vec))
    # else:
    #     print("LOG: Jump 1 (EQ1->EQ2) resulted in no finite equilibrium.")

    # 3. Try Jump 2: EQ2 -> EQ1 direction
    
    final_tableau2 = jump_exploration(game, EQ_list[1], EQ_list[0])
    if final_tableau2:
        print("LOG: Jump 2 (EQ2->EQ1) Result:")
        EQ4_vec = game.getequil(final_tableau2)
        print(format_eq_string(game, EQ4_vec))
    else:
        print("LOG: Jump 2 (EQ2->EQ1) resulted in no finite equilibrium.")
 
    
if __name__ == "__main__":
    explore("poly2.txt", True)