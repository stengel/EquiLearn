import random #TODO use seed
import numpy as np
np.set_printoptions(precision=2, suppress=False)
from agent import *

#TODO: create base game class in a way that abstracts all info from specific games. 
#       > probably would invovle creating "environment", "observation", and "action" types

class DuopolyGame(): #TODO: inheret from base game class
    def __init__(self, T):
        self.T = T
        self.costs = [57, 71]
        self.reset()

    def reset(self):
        self.prices = [[],[]]
        self.profits = [[],[]]
        self.ds = [[200], [200]]

    def single_round(self, p1, p2, t):
        next_price = [0,0]
        next_price[0] = p1.choose_price(self.ds[0][-1], self.costs[0], t)
        next_price[1] = p2.choose_price(self.ds[1][-1], self.costs[1], t)
        for player in [0,1]:
            self.prices[player].append(next_price[player])
            self.profits[player].append((self.ds[player][-1] - self.prices[player][-1]) * (self.prices[player][-1] - self.costs[player]))
        
        self.ds[0].append((self.ds[0][-1] + 0.5*(self.prices[1][-1] - self.prices[0][-1])))
        self.ds[1].append((self.ds[1][-1] + 0.5*(self.prices[0][-1] - self.prices[1][-1])))

    def run(self, p1, p2):
        p1.reset()
        p2.reset()
        self.reset()
        for t in range(self.T-1):
            self.single_round(p1,p2,t)

        #HACK END EFFECT TEST:
        for player in [0,1]:
            self.prices[player].append((self.costs[player] + self.ds[player][-1])*0.5)
            self.profits[player].append((self.ds[player][-1] - self.prices[player][-1]) * (self.prices[player][-1] - self.costs[player]))
        #END HACK

        return self.profits[0], self.profits[1]

def tournament_fixed_length(p1_list, p2_list, T = 25):
    game = DuopolyGame(T)
    A = np.zeros((len(p1_list), len(p2_list)))
    B = np.zeros((len(p1_list), len(p2_list)))
    for i, p1 in enumerate(p1_list):
        for j, p2 in enumerate(p2_list):
            p1_profits, p2_profits = game.run(p1, p2)
            A[i][j] = sum(p1_profits) / T
            B[i][j] = sum(p2_profits) / T
    return A, B


#TESTING Equivalence with Bernhard's play.py
p1_list = [
    Myopic(),
    Guess(125, 207, 61, 125),
    Const(125),
    Const(117),
    Const(114.2),
    Const(105),
    Const(100),
    Const(95),
    Imit(120),
    Imit(110),
    Fight(125, 207)
]

p2_list = [
    Myopic(),
    Guess(130, 193, 75, 130),
    Imit(131),
    Imit(114.2),
    Fight(130, 193)
]

A,B = tournament_fixed_length(p1_list, p2_list, 5)

print(A)


T=5 # number of rounds
# player 0 = low cost
# player 1 = high cost
cost = [57, 71] # cost
# first index is always player
demandpotential = [[0]*T,[0]*T] # two lists for the two players
demandpotential[0][0]=200 # initialize first round 0
demandpotential[1][0]=200
prices = [[0]*T,[0]*T]  # prices over T rounds
profit = [[0]*T,[0]*T]  # profit in each of T rounds

def monopolyprice(player, t): # myopic monopoly price 
    return (demandpotential[player][t] + cost[player])/2

def updatePricesProfitDemand(pricepair, t):
    # pricepair = list of prices for players 0,1 in current round t
    for player in [0,1]:
        price = pricepair[player]
        prices[player][t] = price
        profit[player][t] = \
            (demandpotential[player][t] - price)*(price - cost[player])
        if t<T-1 :
            demandpotential[player][t+1] = \
                demandpotential[player][t] + (pricepair[1-player] - price)/2
    return

def totalprofit(): # gives pair of total profits over T periods
    return sum(profit[0]), sum(profit[1])

def avgprofit(): # gives pair of average profits per round
    return sum(profit[0])/T, sum(profit[1])/T

def match (stra0, stra1):
    # matches two strategies against each other over T rounds
    # each strategy is a function giving price in round t
    # assume demandpotentials in round 0 are untouched, rest
    # will be overwritten
    for t in range(T):
        pricepair = [ stra0(t), stra1(t) ]
        # no dumping
        pricepair[0] = max (pricepair[0], cost[0])
        pricepair[1] = max (pricepair[1], cost[1])
        updatePricesProfitDemand(pricepair, t)
    return avgprofit()

def tournament(strats0, strats1):
    # strats0,strats1 are lists of strategies for players 0,1
    # all matched against each other
    # returns resulting pair A,B of payoff matrices 
    m = len(strats0)
    n = len(strats1)
    # A = np.array([[0.0]*n]*m) # first index=row, second=col
    # B = np.array([[0.0]*n]*m)
    A = np.zeros((m,n))
    B = np.zeros((m,n))
    for i in range (m):
        for j in range (n):
            A[i][j], B[i][j] = match (strats0[i], strats1[j])
    return A,B 

# strategies with varying parameters
def myopic(player, t): 
    return monopolyprice(player, t)    

def const(player, price, t): # constant price strategy
    if t == T-1:
        return monopolyprice(player, t)
    return price

def imit(player, firstprice, t): # price imitator strategy
    if t == 0:
        return firstprice
    if t == T-1:
        return monopolyprice(player, t)
    return prices[1-player][t-1] 

def fight(player, firstprice, t): # simplified fighting strategy
    if t == 0:
        return firstprice
    if t == T-1:
        return monopolyprice(player, t)
    aspire = [ 207, 193 ] # aspiration level for demand potential
    D = demandpotential[player][t] 
    Asp = aspire [player]
    if D >= Asp: # keep price; DANGER: price will never rise
        return prices[player][t-1] 
    # adjust to get to aspiration level using previous
    # opponent price; own price has to be reduced by twice
    # the negative amount D - Asp to get demandpotential to Asp 
    P = prices[1-player][t-1] + 2*(D - Asp) 
    # never price to high because even 125 gives good profits
    P = min(P, 125)
    return P

# sophisticated fighting strategy, compare fight()
# estimate *sales* of opponent as their target, kept between
# calls in global variable oppsaleguess[]. Assumed behavior
# of opponent is similar to this strategy itself.
oppsaleguess = [61, 75] # first guess opponent sales as in monopoly
def guess(player, firstprice, t): # predictive fighting strategy
    if t == 0:
        oppsaleguess[0] = 61 # always same start 
        oppsaleguess[1] = 75 # always same start 
        return firstprice
    if t == T-1:
        return monopolyprice(player, t)
    aspire = [ 207, 193 ] # aspiration level
    D = demandpotential[player][t] 
    Asp = aspire [player]
    if D >= Asp: # keep price, but go slightly towards monopoly if good
        pmono = monopolyprice(player, t)
        pcurrent = prices[player][t-1] 
        if pcurrent > pmono: # shouldn't happen
            return pmono
        if pcurrent > pmono-7: # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-7)
    # guess current *opponent price* from previous sales
    prevsales = demandpotential[1-player][t-1] - prices[1-player][t-1] 
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * oppsaleguess[player] + (1-alpha)*prevsales
    # update
    oppsaleguess[player] = newsalesguess 
    guessoppPrice = 400 - D - newsalesguess 
    P = guessoppPrice + 2*(D - Asp) 
    if player == 0:
        P = min(P, 125)
    if player == 1:
        P = min(P, 130)
    return P

strats0 = [ # use lambda to get function with single argument t
    lambda t : myopic(0,t)        # 0 = myopic
    , lambda t : guess(0,125,t)   # 1 = clever guess strategy
    , lambda t : const(0,125,t)
    , lambda t : const(0,117,t) 
    , lambda t : const(0,114.2,t)
    , lambda t : const(0,105,t) 
    , lambda t : const(0,100,t)   # suppressed for easier plot
    , lambda t : const(0,95,t) 
    , lambda t : imit(0,120,t)
    , lambda t : imit(0,110,t)
    , lambda t : fight(0,125,t)
]

strats1 = [
    lambda t : myopic(1,t)        # 0 = myopic
    , lambda t : guess(1,130,t)   # 1 = clever guess strategy
    , lambda t : imit(1,131,t)    # 2 = imit starting nice
    , lambda t : imit(1,114.2,t)  # 3 = imit starting competitive
    , lambda t : fight(1,130,t)   # 4 = aggressive fight
]

A1,B1 = tournament (strats0, strats1)

print(A1)
print(A1 - A)