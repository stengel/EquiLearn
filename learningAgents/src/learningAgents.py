import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import numpy as np # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)

sys.append("../../05Aug2022")

class DemandPotentialGame():
    """
        Fully defines demand Potential Game. 
    """
    
    def __init__(self, totalDemand, tupleCosts, adversaryPolicy, stages) -> None:
        self.totalDemand = totalDemand
        self.tupleCosts = tupleCosts
        self.T = stages
        # first index is always player
        self.demandPotential = None # two lists for the two players
        self.prices = None # prices over T rounds
        self.profit = None  # profit in each of T rounds
        self.stage = None


    def profits(self, demandPotential, price, agent = 0):
        return (demandPotential - price)*(price - cost[agent])

    def updatePricesProfitDemand(self, pricepair):
        # pricepair = list of prices for players 0,1 in current round t
        for player in [0,1]:
            price = pricepair[player]
            prices[player][t] = price
            profit[player][t] = \
                (demandpotential[player][t] - price)*(price - cost[player])
            if t<T-1 :
                demandpotential[player][t+1] = \
                    demandpotential[player][t] + (pricepair[1-player] - price)/2


    def monopolyPrice(self, player, stage): # myopic monopoly price 
        return (demandpotential[player][t] + cost[player])/2 

    # Adversary strategies with varying parameters
    def myopic(self, player): 
        return monopolyprice(player, t)    

    def const(self, player, price, t): # constant price strategy
        if t == T-1:
            return monopolyprice(player, t)
        return price

    def imit(self, player, firstprice, t): # price imitator strategy
        if t == 0:
            return firstprice
        if t == T-1:
            return monopolyprice(player, t)
        return prices[1-player][t-1] 

    def fight(self, player, firstprice, t): # simplified fighting strategy
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
    def guess(self, player, firstprice, t): # predictive fighting strategy
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


class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined.
    """
    def __init__(self, totalDemand, tupleCosts, transitionFunction, actionSpace, initState) -> None:
        super().__init__()

        self.transitionFunction = transitionFunction
        self.actionSpace = actionSpace
        self.rewardFunction = self.profits
        self.initState = initState
        self.episodesMemory = list()

        self.state = None

        self.done = False

    def reset(self):
        reward = 0
        self.stage = 0
        self.state = self.initState
        self.demandPotential = [[0]*T,[0]*T] # two lists for the two players
        self.prices = [[0]*T,[0]*T]  # prices over T rounds
        self.profit = [[0]*T,[0]*T]  # profit in each of T rounds
        self.demandPotential[0][0] = self.totalDemand/2 # initialize first round 0
        self.demandPotential[1][0] = self.totalDemand/2
        return self.state, reward, self.done


    def adversaryChoosePrice(self) 
        return myopic(1, self.stage)

    def step(self, state, action):
        adversaryAction = self.adversaryPolicy()
        updatePricesProfitDemand(self, pricepair)
        newState = (...)
        reward = profit[0][self.stage]

        return newState, reward, self.stage < self.T





class REINFORCEALgorithm():
    """
        Model Solver.
    """
    def __init__(self, Model, policyNet, numberEpisodes) -> None:
        self.env = Model
        self.env.adversaryReturns = np.zeros(numberEpisodes)
        self.returns = np.zeros(numberEpisodes)
        self.policy = policyNet
        self.numberEpisodes = numberEpisodes

    def  solver(self):

        for episode in range(numberEpisodes):
            episodeMemory = list()
            state, reward, done, _ = self.env.reset()

            while not done:
                prev_state = self.state
                probs = self.policy(prev_state)
                distAction = Categorical(probs)
                action = distAction.sample()

                state, reward, done, _ = env.step(prev_state, action.item())
                
                episodeMemory.append((prev_state, action, reward))
                self.env.stage += 1

            states = torch.stack([item[0] for item in episodeMemory])
            actions = torch.tensor([item[1] for item in episodeMemory])
            rewards = torch.tensor([item[2] for item in episodeMemory])

            action_probs = policy(states)
            action_dists = Categorical(action_probs)
            action_logprobs = action_dists.log_prob(actions)

            returns = returns(reward, episodeMemory)

            loss = - ( torch.sum(returns*action_logprobs) )/len(episodeMemory)

            optim.zero_grad()
            loss.backward()
            optim.step()



    def returns(self, reward, episodeMemory):
        return torch.tensor( [torch.sum( reward[i:]*
        (gamma**torch.arange(i, len(episodeMemory) )) ) for in range(len(episodeMemory))] )
	 
