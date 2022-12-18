# Francisco, Sahar, Edward
# ReinforceAlgorithm Class: Solver.

import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import numpy as np # numerical python
import pandas as pd
from matplotlib import pyplot as plt
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)

class Solver():
    
    def __init__(self,numberEpisodes, Model, discountFactor, numberIterations):
        self.numberEpisodes = numberEpisodes    
        self.env = Model
        self.gamma = discountFactor
        self.numberIterations = numberIterations
        self.bestPolicy=None

    
    def __init__(self,numberEpisodes, Model, discountFactor, numberIterations):
        self.numberEpisodes = numberEpisodes    
        self.env = Model
        self.gamma = discountFactor
        self.numberIterations = numberIterations
        self.bestPolicy=None
        

    def runBestPolicy(self):
        """
            Run best policy from the Reinforcement Learning Algorithm. It needs to be used after training.
        """

        state, reward, done = self.env.reset()
        returns = 0
        while not done:
            prev_state = state
            probs = self.bestPolicy(prev_state)
            distAction = Categorical(probs)
            action = distAction.sample()

            state, reward, done = self.env.step(prev_state, action.item())
            returns = returns + reward
        

        return returns
    


class ReinforceAlgorithm(Solver):
    """
        Model Solver.
    """
    def __init__(self, Model, neuralNet, numberIterations, numberEpisodes, discountFactor) -> None:
        super().__init__(numberEpisodes, Model, discountFactor, numberIterations)

        self.env.adversaryReturns = np.zeros(numberEpisodes)
        self.neuralNetwork = neuralNet  
        self.policy = None
        self.optim = None
        self.bestAverageRetu = 0
        self.returns = np.zeros((numberIterations, numberEpisodes))   


    def resetPolicyNet(self):
        """
            Reset Policy Neural Network.
        """
        self.policy, self.optim = self.neuralNetwork.reset()

    def savePolicy(self):
         pass

    def  solver(self):
        """
            Method that performs Monte Carlo Policy Gradient algorithm. 
        """

        for iteration in range(self.numberIterations):
            self.resetPolicyNet()

            for episode in range(self.numberEpisodes):
                if episode % 50000 == 0:
                    print (episode)
                episodeMemory = list()
                state, reward, done = self.env.reset()
                retu = 0
                while not done:
                    prev_state = state
                    probs = self.policy(prev_state)
                    distAction = Categorical(probs)
                    action = distAction.sample()

                    state, reward, done = self.env.step(prev_state, action.item())
                    retu = retu + reward
                    episodeMemory.append((prev_state, action, reward))
		
		
                states = torch.stack([item[0] for item in episodeMemory])    
                actions = torch.tensor([item[1] for item in episodeMemory]) 
                rewards = torch.tensor([item[2] for item in episodeMemory])
		

                action_probs = self.policy(states) 
                action_dists = Categorical(action_probs) 
                action_logprobs = action_dists.log_prob(actions)

                returns = self.returnsComputation(rewards, episodeMemory)

                loss = - ( torch.sum(returns*action_logprobs) )/len(episodeMemory)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.returns[iteration][episode] = retu #sum of the our player's rewards  rounds 0-25 

                
            averageRetu= ((self.returns[iteration]).sum())/(self.numberEpisodes)
            if (self.bestPolicy is None) or (averageRetu > self.bestAverageRetu):
                self.bestPolicy=self.policy
                self.bestAverageRetu=averageRetu
            


    def returnsComputation(self, rewards, episodeMemory):
        """
        Method computes vector of returns for every stage. The returns are the cumulative rewards from that stage.
        """
        return torch.tensor( [torch.sum( rewards[i:] * (self.gamma ** torch.arange(0, (len(episodeMemory)-i))) ) for i in range(len(episodeMemory)) ] )
	 
#class ActorCriticAlgorithm(Solver):
    
