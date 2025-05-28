#!/usr/bin/python
# file utilities
# tofraction utilities with global decimals

import fractions
import numpy as np

# global constants, mutable
# https://stackoverflow.com/questions/1977362/how-to-create-module-wide-variables-in-python
decimals = 4
deciDenom = 10**decimals
MAXDECIMALS = 20 
# roundingwarn = False

def setdecimals(n):
    global decimals,deciDenom
    if n>=0 and n<=MAXDECIMALS:
        decimals = n
        deciDenom = 10**decimals
    else:
        # if roundingwarn:
        print(n,"as number of decimals not in allowed range 0 to", MAXDECIMALS)
    return

commentchars = "#%*" # lines starting with these are ignored

# read file into list of line-strings
# truncate leading and trailing blanks
# ignore blank lines and lines starting with commentchars
def stripcomments(filename):
    # http://stackoverflow.com/questions/12330522/reading-a-file-without-newlines
    newlist = [] 
    with open(filename,'r') as temp:
        temp = temp.read().splitlines()
        # strip comments
        for line in temp:
            line = line.strip()
            if line != "" and not line[0] in commentchars:
                newlist.append(line)
    return newlist

# convert lines to words
def towords(lines):
    words = []
    for line in lines:
        l = line.split()
        for w in l:
            words.append(w)
    return words

# convert s to fraction
# if s contains ".": convert to decimal fraction
# (numerator deciDenom)
def tofraction(s):
    if isinstance(s, str) and "." in s:
        s = float(s)
    if isinstance(s, float): 
        num = int(abs(s)*deciDenom+0.5) # round .5 away from zero
        if s<0:
            num = -num
        return fractions.Fraction(num,deciDenom)
    # any other s than a float or string containing '.':
    return fractions.Fraction(s)

def matrix_tofraction(A): # incoming A must be a numpy array
    shape = A.shape
    AA = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            AA[i][j] = tofraction(A[i][j])
    return AA

# create n-vector of fractions from words[start,start+n)
def tovector(n, words, start):
    vector = np.zeros( (n), dtype=fractions.Fraction)
    for i in range(n):
        vector[i] = tofraction(words[start+i])
    return vector

# create (m,n)-matrix of fractions from words[start,start+m*n)
def tomatrix(m, n, words, start):
    C = np.zeros( (m,n), dtype=fractions.Fraction)
    k = start
    for i in range(m):
        for j in range(n):
            C[i][j] = tofraction(words[k])
            k+=1
    return C

def fractions_np_to_int(tupl):
    frac = []
    for f in tupl:
        n = f.numerator
        d = f.denominator

        if type(n) != int:
            n = n.item()
        if type(d) !=int:
            d = d.item()
        frac.append(fractions.Fraction(n,d))
    return tuple(frac)