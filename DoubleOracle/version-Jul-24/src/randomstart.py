#!/usr/bin/python

import random
import math 
import fractions
import matplotlib.pyplot as plt
import sys

# give random n-tuple uniformly from unit simplex
def randInSimplex(n, naive=False):
    x = [0.0]*n
    if naive: # random numbers re-normalized
        sum = 0
        for i in range(n):
            x[i] = random.uniform(0, 1)
            sum += x[i]
        return [k/sum for k in x] 

    else: # properly uniformly in simplex
        factor = 1.0
        i = n-1
        while i>0:
            b = random.uniform(0, 1)
            if i==1 or b==1.0 :
                f = 1-b
            else:
                f = math.exp(math.log(1-b)/i)
            x[i] = factor * (1-f)
            factor *= f
            i -= 1
        x[0] = factor
        return x

# round an array <x> of probabilities to fractions with
# denominator <accuracy>
def roundArray(x, accuracy=10000):
    n = len(x)
    sum = 0
    numerator = [0]*n
    pastdecimals = [0.0]*n
    for i in range(n):
        abig = x[i]*accuracy
        num = numerator[i] = int(abig)
        pastdecimals[i] = abig-num
        sum += num
    tobeadded = accuracy - sum
    # print(tobeadded)
    assert (tobeadded >=0 and tobeadded<n), "need probabilities"
    for k in range(tobeadded):
        maxval = max(pastdecimals)
        position = pastdecimals.index(maxval)
        pastdecimals[position] = 0.0
        numerator[position] += 1
    return [fractions.Fraction(k, accuracy) for k in numerator] 

# renormalize list x to sum to one
def renormalize(x):
    s = sum(x)
    if s == 0:
        return x
    return [k/s for k in x]

# map triple of unit triangle to pair in 2D
# with corners [0,0] [1,0] [0.5,sqrt(3)/2]
def maptotriangle(vec):
    x = vec[1] + 0.5 * vec[2]
    y = 3**.5/2 * vec[2]
    return x,y

if __name__ == "__main__":
    arglist = sys.argv
    print("Usage: ",arglist[0],
        "[numpoints [accuracy [higherdim ['n[aive]']]]]")
    numpoints = 200 # number of points plotted
    accuracy = 20   # coarse accuracy
    higherdim = 3   # display middle 3 dimensions
    naiveplot = False # if True just sum random numbers
    if len(arglist)>1:
        numpoints = int(arglist[1])
    if len(arglist)>2:
        accuracy = int(arglist[2])
    if len(arglist)>3:
        a = int(arglist[3])
        if a>2 and a<11:
            higherdim = a
    if len(arglist)>4:
        naiveplot = True
    print (f"numpoints={numpoints} numpoints={numpoints} higherdim={higherdim} naiveplot={naiveplot}")
    if higherdim>3:
        segmentstart = (higherdim-2)//2
        print ("show positions", segmentstart,"..",
            segmentstart+2,"of 0 ..", higherdim-1)
    fig1, ax = plt.subplots()
    ax.set_box_aspect(.866)
    # plt.axis('square')
    x1,y1 = maptotriangle([1,0,0])
    x2,y2 = maptotriangle([0,1,0])
    x3,y3 = maptotriangle([0,0,1])
    plt.plot ([x1,x2,x3,x1], [y1,y2,y3,y1], "black") 

    roundedpoints = []
    for i in range(numpoints):
        point = randInSimplex(higherdim, naiveplot)
        if higherdim>3:
            segmentstart = (higherdim-2)//2
            point = renormalize (point[segmentstart:segmentstart+3])
        roundedpoints.append(roundArray(point,accuracy)) 
        x,y = maptotriangle(point)
        plt.plot([x], [y], 'g.')
    for circ in roundedpoints:
        x,y = maptotriangle(circ)
        plt.scatter([x], [y], s=10000//accuracy, facecolors='none',
            edgecolors='r')
    plt.show() 
