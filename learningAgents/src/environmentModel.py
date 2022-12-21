# Francisco, Sahar, Edward
# Contains Game Class and Model of the Game Class.

from enum import Enum
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys # Not used?
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
        """
        Method resets game memory: Demand Potential, prices, profits
        """
        self.demandPotential = [[0]*(self.T),[0]*(self.T)] # two lists for the two players
        self.prices = [[0]*self.T,[0]*self.T]  # prices over T rounds
        self.profit = [[0]*self.T,[0]*self.T]  # profit in each of T rounds
        self.demandPotential[0][0] = self.totalDemand/2 # initialize first round 0
        self.demandPotential[1][0] = self.totalDemand/2


    def profits(self, player = 0):
        """
        Computes profits. Player 0 is the learning agent.
        """
        return self.profit[player][self.stage]

    def updatePricesProfitDemand(self, pricePair):
        """
        Updates Prices, Profit and Demand Potential Memory.
        Parameters. 
        pricePair: Pair of prices from the Learning agent and adversary.
        """

        for player in [0,1]:
            price = pricePair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (self.demandPotential[player][self.stage] - price)*(price - self.costs[player])
            if self.stage < self.T-1 :
                self.demandPotential[player][ self.stage + 1] = \
                    self.demandPotential[player][self.stage] + (pricePair[1-player] - price)/2


    def monopolyPrice(self, player, t): # myopic monopoly price 
        """
            Computes Monopoly prices.
        """
        return (self.demandPotential[player][self.stage] + self.costs[player])/2 

    def myopic(self, player = 0): 
        """
            Adversary follows Myopic strategy
        """
        return self.monopolyPrice(player, self.stage)    

    def const(self, player, price): # constant price strategy
        """
            Adversary follows Constant strategy
        """    
        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        return price

    def imit(self, player, firstprice): # price imitator strategy
        if self.stage == 0:
            return firstprice
        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        return self.prices[1-player][self.stage-1] 

    def fight(self, player, firstprice): # simplified fighting strategy
        if self.stage == 0:
            return firstprice
        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        #aspire = [ 207, 193 ] # aspiration level for demand potential
        aspire=[0,0]
        for i in range(2):
            aspire[i]= (self.totalDemand-self.costs[player]+self.costs[1-player])/2
        
        D =self.demandPotential[player][self.stage] 
        Asp = aspire [player]
        if D >= Asp: # keep price; DANGER: price will never rise
            return self.prices[player][self.stage-1] 
        # adjust to get to aspiration level using previous
        # opponent price; own price has to be reduced by twice
        # the negative amount D - Asp to getself.demandPotential to Asp 
        P = self.prices[1-player][self.stage-1] + 2*(D - Asp) 
        # never price to high because even 125 gives good profits
        # P = min(P, 125)
        aspire_price= (self.totalDemand+self.costs[0]+self.costs[1])/4
        P= min(P, int(0.95*aspire_price))

        return P

    def fight_lb(self, player, firstprice):
        P=self.fight(player,firstprice)
        # never price less than production cost
        P=max(P, self.costs[player])
        return P

    # sophisticated fighting strategy, compare fight()
    # estimate *sales* of opponent as their target, kept between
    # calls in global variable oppsaleguess[]. Assumed behavior
    # of opponent is similar to this strategy itself.
    oppsaleguess = [61, 75] # first guess opponent sales as in monopoly
    def guess(self, player, firstprice): # predictive fighting strategy
        if self.stage == 0:
            self.oppsaleguess[0] = 61 # always same start 
            self.oppsaleguess[1] = 75 # always same start 
            return firstprice

        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        aspire = [ 207, 193 ] # aspiration level
        D =self.demandPotential[player][self.stage] 
        Asp = aspire [player]

        if D >= Asp: # keep price, but go slightly towards monopoly if good
            pmono = self.monopolyPrice(player, self.stage)
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
        newsalesguess = alpha * self.oppsaleguess[player] + (1-alpha)*prevsales
        # update
        self.oppsaleguess[player] = newsalesguess 
        guessoppPrice = 400 - D - newsalesguess 
        P = guessoppPrice + 2*(D - Asp) 
        
        if player == 0:
            P = min(P, 125)
        if player == 1:
            P = min(P, 130)
        return P    


class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined. The class is a Child from the Demand Potential Game Class.
        The reason: Model is a conceptualization of the Game.
    """
    def __init__(self, totalDemand,tupleCosts,totalStages,adversaryProbs, advHistoryNum=3) -> None:
        super().__init__( totalDemand,tupleCosts,totalStages)

        self.rewardFunction = self.profits

        # [stage, agent's demand potential, agent's last price, history of adversary's prices]
        self.initState = [0, totalDemand/2, ((totalDemand/2) + tupleCosts[0])/2] + ([0]*advHistoryNum) 
        self.episodesMemory = list()
        self.done = False
        self.adversaryProbs=adversaryProbs
        self.advHistoryNum=advHistoryNum #number of previous adversary's action we consider in the state
        
    def reset(self):
        """
            Reset Model Instantiation. 
        """
        reward = 0
        self.stage = 0
        self.done=False
        self.resetGame()
        self.resetAdversary()
        return torch.tensor(self.initState, dtype=torch.float32), reward, self.done

    def resetAdversary(self):
        adversaryDist= Categorical(self.adversaryProbs)
        adversaryInd = (adversaryDist.sample()).item()
        self.adversaryMode=AdversaryModes(adversaryInd)
        # print(self.adversaryMode)

    def adversaryChoosePrice(self): 
        """
            Strategy followed by the adversary.
        """

        if self.adversaryMode== AdversaryModes.constant_132:
            return self.const(player=1,price=132)
        elif self.adversaryMode== AdversaryModes.constant_95:
            return self.const(player=1,price=95)
        elif self.adversaryMode== AdversaryModes.imitation_128:
            return self.imit(player=1,firstprice=128)
        elif self.adversaryMode== AdversaryModes.imitation_132:
            return self.imit(player=1,firstprice=132)
        elif self.adversaryMode== AdversaryModes.fight_100:
            return self.fight(player=1,firstprice=100)
        elif self.adversaryMode== AdversaryModes.fight_125:
            return self.fight(player=1,firstprice=125)
        elif self.adversaryMode== AdversaryModes.fight_lb_125:
            return self.fight_lb(player=1,firstprice=125)
        elif self.adversaryMode== AdversaryModes.fight_132:
            return self.fight(player=1,firstprice=132)
        elif self.adversaryMode== AdversaryModes.fight_lb_132:
            return self.fight_lb(player=1,firstprice=132)
        elif self.adversaryMode== AdversaryModes.guess_125:
            return self.fight(player=1,firstprice=125)
        elif self.adversaryMode== AdversaryModes.guess_132:
            return self.fight(player=1,firstprice=132)
        else:
            return self.myopic(player = 1)





    def step(self, state, action):
        """
        Transition Function. 
        Parameters:
        - action: Price
        - state: tupple in the latest stage (Demand Potential, Price)
        """
        adversaryAction = self.adversaryChoosePrice()
        myopicPrice=self.myopic()
        self.updatePricesProfitDemand( [self.myopic() - action, adversaryAction] )     

        done= (self.stage == self.T-1)

        advHistory= [0]*self.advHistoryNum
        j=self.advHistoryNum-1
        for i in range(self.stage, max(-1, self.stage-self.advHistoryNum),-1):
            advHistory[j]=self.prices[1][i]
            j-=1


        if not done:
            newState = [self.stage+1 ,self.demandPotential[0][self.stage + 1], self.prices[0][self.stage] ] + advHistory
        else:
            newState=[self.stage+1 ,0, self.prices[0][self.stage] ] + advHistory
        
        reward = self.rewardFunction()
        self.stage = self.stage + 1

        
        return torch.tensor(newState, dtype=torch.float32), reward, done

class AdversaryModes(Enum):
    myopic=0
    constant_132=1
    constant_95=2
    imitation_132=3
    imitation_128=4
    fight_132=5
    fight_lb_132=6
    fight_125=7
    fight_lb_125=8
    fight_100=9
    guess_132=10
    guess_125=11
    
    
    
    
