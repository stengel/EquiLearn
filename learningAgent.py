# Ed, Kat, Galit
# ReinforceAlgorithm Class: Solver.

import numpy as np #repeated
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)

5
class ReinforceAlgorithm():
    """
        Model Solver.
    """
    #def __init__(self, Model, neuralNet, numberIterations, numberEpisodes, discountFactor) -> None:
    def __init__(self, Model, Qtable, numberEpisodes, discountFactor) -> None:
        self.env = Model
        self.env.adversaryReturns = np.zeros(numberEpisodes)
        self.returns = np.zeros( numberEpisodes)
        self.numberEpisodes = numberEpisodes
        self.episodesMemory = list()
        self.gamma = discountFactor
        self.Qtable=Qtable
        self.numStates=len(Qtable) #Qtable dimen - number of states x number of actions
        if self.numStates % 2 ==1 :
            print('num of states should be even')
        self.numActions=len(Qtable[0])
        if self.numActions % 2 ==1 :
            print('num of actions should be even')
        #self.numberIterations = numberIterations
        #self.neuralNetwork = neuralNet
        #self.policy = None
        #self.optim = None
        #self.bestPolicy=None
        #self.bestAverageRetu = 0

        """
        This function will eventually choose and adversary agent to play against
        and according to that return an action played by that adversary. 
        This could be chosen according to some existing equilibruim dist.
        """
    def chooseAdver(self, st):
        return (400-st +self.env.costs[1])/2  #myopic

    def StateInd(self, state):                 # computes state index in Qtable
        return state -(200-self.numStates/2)

    def ActionInd(self, monprice, action):      # computes action index in Qtable
        return action-(monprice-self.numActions+1)

    def alpha_n(self, n):                       # defines the Qlearning rate
        return 1/n
    """
    Now the Q-learning itself 
    """
    def  solver(self):
        for episode in range(self.numberEpisodes):
            #state, reward, done = self.env.reset()
            state=np.random.randint(200-(self.numStates)/2,200+(self.numStates)/2)
            monprice=(state + self.env.costs[0]) / 2
            action=np.random.randint(monprice-self.numActions+1, monprice+1)
            advAction=self.chooseAdver(state)
            reward=(state-action)*(action-self.env.costs[0])
            next_state=state+.5*(advAction-action)
            opt_value_next=max(self.Qtable[self.StateInd(next_state)])

            # updating the Qtable
            qvalue=(1-self.alpha_n(episode))*\
                   self.Qtable[self.StateInd(state),self.ActionInd(monprice, action)]\
                   +self.alpha_n(episode)*\
                   (reward+self.gamma*opt_value_next)
            self.Qtable[self.StateInd(state),self.ActionInd(monprice, action)]=qvalue


