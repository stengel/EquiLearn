#cython interface for lrsnash.c

cimport cython
import numpy as np
cimport numpy as np

import sympy as sp
from fractions import Fraction

import warnings

MAXSTRAT = 200 #used for padding. should match lrsnash.h
MAXLONG = 2147483647 #maxmimum a long can be in C

cdef extern from "lib/lrsnash.h": 
    ctypedef struct ratnum:
        long num
        long den
    ctypedef struct game:
        long nstrats[2]
        ratnum payoff[200][200][2] #ctypedef doesn't allow variables for lengths
    ctypedef struct mixedstrat:
        long nstrats
        ratnum weights[200]
    ctypedef struct equilibrium:
        mixedstrat strats[2]
    equilibrium lrs_solve_nash(game *g)

def solve_nash(A, B, maxden=1000):
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("Payoffs must be 2 dimensional arrays")
    if A.shape != B.shape:
        raise ValueError("Shapes of payoff matrices much match")

    #convert A and B to array of ratnums
    def frac_to_ratnum(frac):
        cdef ratnum r
        if frac.numerator > MAXLONG:
            raise ValueError("Numerator of fraction must be less than " + str(MAXLONG))
        r.num = frac.numerator
        r.den = frac.denominator
        return r
    frac_to_ratnum_array = np.vectorize(frac_to_ratnum)

    def int_to_ratnum(i):
        cdef ratnum r
        r.num = i
        r.den = 1
        return r
    int_to_ratnum_array = np.vectorize(int_to_ratnum)

    def float_to_ratnum(f):
        frac = sp.Rational(f).limit_denominator(maxden)
        return frac_to_ratnum(frac)
    float_to_ratnum_array = np.vectorize(float_to_ratnum)

    has_warned = False
    if A.dtype.kind == "O" and type(A[0][0]) in [type(Fraction(0.1)), type(sp.Rational(0.1))]:
        A = frac_to_ratnum_array(A)
    elif A.dtype.kind in "ui":
        A = int_to_ratnum_array(A)
    elif A.dtype.kind == "f":
        warnings.warn("Floating point numbers will be rounded to rationals with "\
                        "maximum denominator equal to \"maxnum\" argument "\
                    "(default 1000). This may lead to inconsitent results.")
        has_warned = True
        A = float_to_ratnum_array(A)
    else:
        raise TypeError("Payoff matrices must be arrays of type Fraction, sp.Rational, integer, or float")

    if B.dtype.kind == "O" and type(B[0][0]) in [type(Fraction(0.1)), type(sp.Rational(0.1))]:
        B = frac_to_ratnum_array(B)
    elif B.dtype.kind in "ui":
        B = int_to_ratnum_array(B)
    elif B.dtype.kind == "f":
        if not has_warned:
            warnings.warn("Floating point numbers will be rounded to rationals with "\
                            "maximum denominator equal to \"maxnum\" argument "\
                            "(default 1000). This may lead to inconsitent results.")
        B = float_to_ratnum_array(B)
    else:
        raise TypeError("Payoff matrices must be arrays of type Fraction, sp.Rational, integer, or float")
        
    # pad payoffs to be of maximum size
    padded_A = int_to_ratnum_array(np.zeros((MAXSTRAT, MAXSTRAT)))
    padded_B = int_to_ratnum_array(np.zeros((MAXSTRAT, MAXSTRAT)))
    padded_A[:A.shape[0],:A.shape[1]] = A
    padded_B[:B.shape[0],:B.shape[1]] = B

    #define game and find equilibria using c library
    cdef game g
    g.nstrats = A.shape
    g.payoff = np.stack((padded_A, padded_B), axis=2).swapaxes(0, 1) #check if need to swap axes

    e = lrs_solve_nash(&g)

    #convert equilibria to sympy-numpy    
    e_A = np.empty(e.strats[0].nstrats,dtype='O')
    for i, r in enumerate(e.strats[0].weights):
        if i == e_A.shape[0]:
            break
        e_A[i] = sp.Rational(r['num'], r['den'])

    e_B = np.empty(e.strats[1].nstrats,dtype='O')
    for i, r in enumerate(e.strats[1].weights):
        if i == e_B.shape[0]:
            break
        e_B[i] = sp.Rational(r['num'], r['den'])

    return (e_A, e_B)
    