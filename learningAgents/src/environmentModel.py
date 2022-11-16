import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import numpy as np # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)



class DemandPotentialGame():
    """
        Fully defines demand Potential Game. It contains game rules, memory and agents strategies.
    """
    
    def __init__(self, totalDemand, tupleCosts, totalStages) -> None:
        self.totalDemand = totalDemand
        self.costs = tupleCosts
        self.T = totalStages
        # first index is always player
        self.demandPotential = None # two lists for the two players
        self.prices = None # prices over T rounds
        self.profit = None  # profit in each of T rounds
        self.stage = None


    def resetGame(self):
        self.demandPotential = [[0]*self.T,[0]*self.T] # two lists for the two players
        self.prices = [[0]*self.T,[0]*self.T]  # prices over T rounds
        self.profit = [[0]*self.T,[0]*self.T]  # profit in each of T rounds
        self.demandPotential[0][0] = self.totalDemand/2 # initialize first round 0
        self.demandPotential[1][0] = self.totalDemand/2


    def profits(self, player = 0):
        return self.profit[player][self.stage]

    def updatePricesProfitDemand(self, pricepair):
        # pricepair = list of prices for players 0,1 in current round t
        for player in [0,1]:
            price = pricepair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (self.demandPotential[player][self.stage] - price)*(price - self.costs[player])
            if self.stage < self.T-1 :
                self.demandPotential[player][ self.stage + 1] = \
                    self.demandPotential[player][self.stage] + (pricepair[1-player] - price)/2


    def monopolyPrice(self, player, t): # myopic monopoly price 
        return (self.demandPotential[player][self.stage] + self.costs[player])/2 

    def myopic(self, player = 0): 
        return self.monopolyPrice(player, self.stage)    

    def const(self, player, price): # constant price strategy
        if self.stage == T-1:
            return monopolyPrice(player, self.stage)
        return price

    def imit(self, player, firstprice): # price imitator strategy
        if self.stage == 0:
            return firstprice
        if self.stage == T-1:
            return monopolyPrice(player, self.stage)
        return self.prices[1-player][t-1] 

    def fight(self, player, firstprice): # simplified fighting strategy
        if self.stage == 0:
            return firstprice
        if self.stage == T-1:
            return monopolyPrice(player, self.stage)
        aspire = [ 207, 193 ] # aspiration level for demand potential
        D =self.demandPotential[player][self.stage] 
        Asp = aspire [player]
        if D >= Asp: # keep price; DANGER: price will never rise
            return self.prices[player][self.stage-1] 
        # adjust to get to aspiration level using previous
        # opponent price; own price has to be reduced by twice
        # the negative amount D - Asp to getself.demandPotential to Asp 
        P = self.prices[1-player][self.stage-1] + 2*(D - Asp) 
        # never price to high because even 125 gives good profits
        P = min(P, 125)
        return P


    # sophisticated fighting strategy, compare fight()
    # estimate *sales* of opponent as their target, kept between
    # calls in global variable oppsaleguess[]. Assumed behavior
    # of opponent is similar to this strategy itself.
    oppsaleguess = [61, 75] # first guess opponent sales as in monopoly
    def guess(self, player, firstprice): # predictive fighting strategy
        if self.stage == 0:
            oppsaleguess[0] = 61 # always same start 
            oppsaleguess[1] = 75 # always same start 
            return firstprice

        if self.stage == T-1:
            return monopolyPrice(player, self.stage)
        aspire = [ 207, 193 ] # aspiration level
        D =self.demandPotential[player][self.stage] 
        Asp = aspire [player]

        if D >= Asp: # keep price, but go slightly towards monopoly if good
            pmono = monopolyPrice(player, self.stage)
            pcurrent = self.prices[player][self.stage-1] 
            if pcurrent > pmono: # shouldn't happen
                return pmono
            if pcurrent > pmono-7: # no change
                return pcurrent
            # current low price at 60%, be accommodating towards "collusion"
            return .6 * pcurrent + .4 * (pmono-7)

        # guess current *opponent price* from previous sales
        prevsales =self.demandPotential[1-player][t-1] - self.prices[1-player][t-1] 
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


class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined.
    """
    def __init__(self, totalDemand,tupleCosts,totalStages, initState) -> None:
        super().__init__( totalDemand,tupleCosts,totalStages)

        self.rewardFunction = self.profits
        self.initState = initState
        self.episodesMemory = list()
        self.done = False

    def reset(self):
        reward = 0
        self.stage = 0
        self.resetGame()
        return torch.tensor(self.initState, dtype=torch.float32), reward, self.done


    def adversaryChoosePrice(self): 
        return self.myopic(player = 1)


    def step(self, state, action):
        adversaryAction = self.adversaryChoosePrice()
        self.updatePricesProfitDemand( [self.myopic() - action, adversaryAction] )
        newState = [ self.demandPotential[1][self.stage + 1], self.prices[1][self.stage] ]
        reward = self.rewardFunction()
        self.stage = self.stage + 1

        
        return torch.tensor(newState, dtype=torch.float32), reward, self.stage == self.T-1

 
