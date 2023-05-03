# Francisco, Sahar, Edward
# Contains Game Class and Model of the Game Class.

from enum import Enum
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys  # Not used?
import numpy as np  # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class DemandPotentialGame():
    """
        Fully defines demand Potential Game. It contains game rules, memory and agents strategies.
    """

    def __init__(self, totalDemand, tupleCosts, totalStages,actionStep=1) -> None:
        self.totalDemand = totalDemand
        self.costs = tupleCosts
        self.T = totalStages
        # first index is always player
        self.demandPotential = None  # two lists for the two players
        self.prices = None  # prices over T rounds
        self.profit = None  # profit in each of T rounds
        self.stage = None
        self.actionStep=actionStep

    def resetGame(self):
        """
        Method resets game memory: Demand Potential, prices, profits
        """
        self.demandPotential = [
            [0]*(self.T+1), [0]*(self.T+1)]  # two lists for the two players
        self.prices = [[0]*self.T, [0]*self.T]  # prices over T rounds
        self.profit = [[0]*self.T, [0]*self.T]  # profit in each of T rounds
        self.demandPotential[0][0] = self.totalDemand / \
            2  # initialize first round 0
        self.demandPotential[1][0] = self.totalDemand/2

        self.our_target_demand=((self.totalDemand+ self.costs[1]-self.costs[0])/2) #target demand
        self.target_price=(self.our_target_demand+self.costs[0])/2

    def profits(self, player=0):
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

        for player in [0, 1]:
            price = pricePair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (
                self.demandPotential[player][self.stage] - price)*(price - self.costs[player])
            if self.stage < self.T-1:
                self.demandPotential[player][self.stage + 1] = \
                    self.demandPotential[player][self.stage] + \
                    (pricePair[1-player] - price)/2

    def monopolyPrice(self, player, t):  # myopic monopoly price
        """
            Computes Monopoly prices.
        """
        return (self.demandPotential[player][self.stage] + self.costs[player])/2

    def myopic(self, player=0):
        """
            Adversary follows Myopic strategy
        """
        return self.monopolyPrice(player, self.stage)

    def const(self, player, price):  # constant price strategy
        """
            Adversary follows Constant strategy
        """
        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        return price

    def imit(self, player, firstprice):  # price imitator strategy
        if self.stage == 0:
            return firstprice
        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        return self.prices[1-player][self.stage-1]

    def fight(self, player, firstprice):  # simplified fighting strategy
        if self.stage == 0:
            return firstprice
        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        # aspire = [ 207, 193 ] # aspiration level for demand potential
        aspire = [0, 0]
        for i in range(2):
            aspire[i] = (self.totalDemand-self.costs[player] +
                         self.costs[1-player])/2

        D = self.demandPotential[player][self.stage]
        Asp = aspire[player]
        if D >= Asp:  # keep price; DANGER: price will never rise
            return self.prices[player][self.stage-1]
        # adjust to get to aspiration level using previous
        # opponent price; own price has to be reduced by twice
        # the negative amount D - Asp to getself.demandPotential to Asp
        P = self.prices[1-player][self.stage-1] + 2*(D - Asp)
        # never price to high because even 125 gives good profits
        # P = min(P, 125)
        aspire_price = (self.totalDemand+self.costs[0]+self.costs[1])/4
        P = min(P, int(0.95*aspire_price))

        return P

    def fight_lb(self, player, firstprice):
        P = self.fight(player, firstprice)
        # never price less than production cost
        P = max(P, self.costs[player])
        return P

    # sophisticated fighting strategy, compare fight()
    # estimate *sales* of opponent as their target

    def guess(self, player, firstprice):  # predictive fighting strategy
        if self.stage == 0:
            self.aspireDemand = [(self.totalDemand+ self.costs[1]-self.costs[0] ), (self.totalDemand+ self.costs[0]-self.costs[1] )]  # aspiration level
            self.aspirePrice = (self.totalDemand+self.costs[0]+self.costs[1])/4
            self.saleGuess= [self.aspireDemand[0]-self.aspirePrice,self.aspireDemand[1]-self.aspirePrice ] # first guess opponent sales as in monopoly ( sale= demand-price)

            return firstprice

        if self.stage == self.T-1:
            return self.monopolyPrice(player, self.stage)
        


        D = self.demandPotential[player][self.stage]
        Asp = self.aspireDemand[player]

        if D >= Asp:  # keep price, but go slightly towards monopoly if good
            pmono = self.monopolyPrice(player, self.stage)
            pcurrent = self.prices[player][self.stage-1]
            if pcurrent > pmono:  # shouldn't happen
                return pmono
            elif pcurrent > pmono-7:  # no change
                return pcurrent
            # current low price at 60%, be accommodating towards "collusion"
            return .6 * pcurrent + .4 * (pmono-7)

        # guess current *opponent price* from previous sales
        prevsales = self.demandPotential[1 -
                                         player][self.stage-1] - self.prices[1-player][self.stage-1]
        # adjust with weight alpha from previous guess
        alpha = .5
        newsalesguess = alpha * self.saleGuess[player] + (1-alpha)*prevsales
        # update
        self.saleGuess[player] = newsalesguess
        guessoppPrice = self.totalDemand - D - newsalesguess
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

    def __init__(self, totalDemand, tupleCosts, totalStages, adversaryProbs, advHistoryNum=3, advNN=None,actionStep=1) -> None:
        super().__init__(totalDemand, tupleCosts, totalStages,actionStep)
        """
        if the adversary is NN then advNN should be set, action step is assumed to be the same as self
        """
        self.rewardFunction = self.profits

        # [stage, agent's demand potential, agent's last price, history of adversary's prices]
        self.episodesMemory = list()
        self.done = False
        self.adversaryProbs = adversaryProbs
        # number of previous adversary's action we consider in the state
        self.advHistoryNum = advHistoryNum
    
        self.advNN=advNN

    def reset(self):
        """
            Reset Model Instantiation. 
        """
        reward = 0
        self.stage = 0
        self.done = False
        self.resetGame()
        self.resetAdversary()
        return self.getState(0), reward, self.done

    def resetAdversary(self):
        adversaryDist = Categorical(self.adversaryProbs)
        adversaryInd = (adversaryDist.sample()).item()
        self.adversaryMode = AdversaryModes(adversaryInd)
        # print(self.adversaryMode)

    def adversaryChoosePrice(self):
        """
            Strategy followed by the adversary.
        """

        if self.adversaryMode == AdversaryModes.constant_132:
            return self.const(player=1, price=132)
        elif self.adversaryMode == AdversaryModes.constant_95:
            return self.const(player=1, price=95)
        elif self.adversaryMode == AdversaryModes.imitation_128:
            return self.imit(player=1, firstprice=128)
        elif self.adversaryMode == AdversaryModes.imitation_132:
            return self.imit(player=1, firstprice=132)
        elif self.adversaryMode == AdversaryModes.fight_100:
            return self.fight(player=1, firstprice=100)
        elif self.adversaryMode == AdversaryModes.fight_125:
            return self.fight(player=1, firstprice=125)
        elif self.adversaryMode == AdversaryModes.fight_lb_125:
            return self.fight_lb(player=1, firstprice=125)
        elif self.adversaryMode == AdversaryModes.fight_132:
            return self.fight(player=1, firstprice=132)
        elif self.adversaryMode == AdversaryModes.fight_lb_132:
            return self.fight_lb(player=1, firstprice=132)
        elif self.adversaryMode == AdversaryModes.guess_125:
            return self.fight(player=1, firstprice=125)
        elif self.adversaryMode == AdversaryModes.guess_132:
            return self.fight(player=1, firstprice=132)
        else:
            return self.myopic(player=1)

    def step(self, state, action,mode=1):
        """
        Transition Function. 
        Parameters:
        - action: Price
        - state: list in the latest stage (stage ,Demand Potential, Agent's Price, Adversary's price hisotry)
        """
        if self.advNN is None:
            adversaryAction = self.adversaryChoosePrice()
            # myopicPrice = self.myopic()
            self.updatePricesProfitDemand(
                [self.myopic() - (self.actionStep* action), adversaryAction])
        else:
            advState= self.getState(self.stage,player=1)
            normAdvState=self.normalizeState(advState,mode=mode)
            probs= self.advNN.policy(normAdvState)
            distAction = Categorical(probs)
            advAction = distAction.sample()
            self.updatePricesProfitDemand(
                [self.myopic() - (self.actionStep* action), self.myopic(player=1)-(self.actionStep*advAction)])
            

        done = (self.stage == self.T-1)

        reward = self.rewardFunction()
        self.stage +=1

        return self.getState(self.stage), reward, done
    
    def getState(self,stage,player=0):
        #[one-hote encoding of stage, our demand, our price, adversary's price history]
        # player=1 is the state for adversary, when we play against another NN, so advNN must be set

        stageEncode=[0]*self.T
        if stage<self.T:
            stageEncode[stage]=1

        
        if stage==0:
            if player ==0:
                state= stageEncode+[self.totalDemand/2, ((self.totalDemand/2) + self.costs[player])/2] + ([0]*self.advHistoryNum)
            else:
                state= stageEncode+[self.totalDemand/2, ((self.totalDemand/2) + self.costs[player])/2] + ([0]*self.advNN.adv_hist)

        else:
            #check last stageeee demand
            state = stageEncode+[self.demandPotential[player][stage], self.prices[player][stage-1]] 
            if(player==0 and self.advHistoryNum>0):
                advHistory = [0]*self.advHistoryNum
                j = self.advHistoryNum-1
                for i in range(stage-1, max(-1, stage-1-self.advHistoryNum), -1):
                    advHistory[j] = self.prices[1-player][i]
                    j -= 1
                state+=advHistory
            elif(player==1 and self.advNN.adv_hist>0):
                advHistory = [0]*self.advNN.adv_hist
                j = self.advNN.adv_hist-1
                for i in range(stage-1, max(-1, stage-1-self.advNN.adv_hist), -1):
                    advHistory[j] = self.prices[1-player][i]
                    j -= 1
                state+=advHistory

        return torch.tensor(state, dtype=torch.float32)
    def normalizeState(self, state, mode=1):
        # [stage one-hot encoded, agent's demand potential, agent's last price, history of adversary's prices]

        if mode == 1:
            normalized = [0]*len(state)
            for i in range(self.T):
                normalized[i] = state[i]
            for i in range(self.T, len(state)):
                normalized[i] = state[i]/(self.totalDemand)
            return torch.tensor(normalized)
        elif mode == 2:

            normalized = torch.zeros(len(state))
            for i in range(self.T):
                normalized[i] = state[i]

            normalized[self.T] = -self.our_target_demand + \
                state[self.T]  # demand

            for i in range(self.T+1, len(state)):
                normalized[i] = -(self.target_price) + \
                    state[i]  # both players' prices
            return normalized
        elif mode == 3:
            return nn.functional.normalize(state, p=2.0, dim=0)
        elif mode == 4:
            normalized = [0]*len(state)
            for i in range(self.T):
                normalized[i] = state[i]
            for i in range(self.T, len(state)):
                normalized[i] = (state[i]-self.costs[0]) / \
                    (self.totalDemand)
            return torch.tensor(normalized)


class AdversaryModes(Enum):
    myopic = 0
    constant_132 = 1
    constant_95 = 2
    imitation_132 = 3
    imitation_128 = 4
    fight_132 = 5
    fight_lb_132 = 6
    fight_125 = 7
    fight_lb_125 = 8
    fight_100 = 9
    guess_132 = 10
    guess_125 = 11
    
