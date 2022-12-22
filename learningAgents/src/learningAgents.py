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
        self.returns = np.zeros((numberIterations, numberEpisodes))   


    def resetPolicyNet(self):
        """
            Reset Policy Neural Network.
        """
        self.policy, self.optim = self.neuralNetwork.reset()
    

    def  solver(self):
        """
            Method that performs Monte Carlo Policy Gradient algorithm. 
        """ 

        for iteration in range(self.numberIterations):
            self.resetPolicyNet()
            
            for episode in range(self.numberEpisodes):
                if episode % 50000 == 0:
                    print(episode)
                episodeMemory = list()
                state, reward, done = self.env.reset()
                
                normState = torch.tensor([  0.0000, 0.0000, 0.0000])
                normState[0] = state[0]/25
                normState[1] = state[1]/400
                normState[2] = state[2]/400                 
                retu = 0
                
                while not done:
                    prevState = state
                    normPrevState = normState                                                        
                    
                    probs = self.policy(normPrevState)
                    distAction = Categorical(probs)
                    action = distAction.sample()
                    

                    state, reward, done = self.env.step(prevState, action.item())
                    normState = torch.tensor([  0.0000, 0.0000, 0.0000])
                    normState[0] = state[0]/25
                    normState[1] = state[1]/400
                    normState[2] = state[2]/400
                    retu = retu + reward
                    episodeMemory.append((normPrevState, action, reward))


                states = torch.stack([item[0] for item in episodeMemory])    
                actions = torch.tensor([item[1] for item in episodeMemory])
                if episode % 5000 == 0:
                    print(actions)
                rewards = torch.tensor([item[2] for item in episodeMemory])

                action_probs = self.policy(states) 
                action_dists = Categorical(action_probs) 
                action_logprobs = action_dists.log_prob(actions)

                returns = self.returnsComputation(rewards, episodeMemory)
                
                if episode % 5000 == 0:
                    print(returns[0])
                    
                loss = - ( torch.sum(returns*action_logprobs) )/len(episodeMemory)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.returns[iteration][episode] = retu #sum of the our player's rewards  rounds 0-25 
                # if we want to look at the discounted returns, we want 
                self.returns[iteration][episode] = returns[0]
       


    def returnsComputation(self, rewards, episodeMemory):
        """
        Method computes vector of returns for every stage. The returns are the cumulative rewards from that stage.
        """
        return torch.tensor( [torch.sum( rewards[i:] * (self.gamma ** torch.arange(0, (len(episodeMemory)-i))) ) for i in range(len(episodeMemory)) ] )

    
