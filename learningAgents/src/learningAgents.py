# Francisco, Sahar, Edward
# ReinforceAlgorithm Class: Solver.

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np # numerical python
import pandas as pd
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)
import matplotlib.pyplot as plt



class Solver():

    
    def __init__(self,numberEpiPerBatch, Model, discountFactor, creditFactor, numberIterations, numberBatches):
        self.numberEpiPerBatch = numberEpiPerBatch    
        self.env = Model
        self.delta = discountFactor # Defined by how patient the player is
        self.gamma = creditFactor # Defined by how much credit we give previous actions
        self.numberIterations = numberIterations
        self.numberBatches = numberBatches
        
        

class ReinforceAlgorithm(Solver):
    """
        Model Solver.
    """
    def __init__(self, Model, neuralNet, numberIterations, numberEpiPerBatch, numberBatches, discountFactor, creditFactor) -> None:
        super().__init__(numberEpiPerBatch, Model, discountFactor, creditFactor, numberIterations, numberBatches)

        self.neuralNetwork = neuralNet  
        self.policy = None
        self.optim = None
        self.returns = np.zeros((numberIterations, numberBatches)) 


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
            
            x = [0]*self.numberBatches
            y = [0]*self.numberBatches            
            
            for batch in range(self.numberBatches):
                earlyExit = True
                totalReturn = 0
                batchStates = torch.empty(0)
                batchActions = torch.empty(0)
                batchRewards = torch.empty(0)
                for episode in range(self.numberEpiPerBatch):
                    discount = 1 / self.delta                
                    episodeMemory = list()
                    state, reward, done = self.env.reset()
                
                    normState = torch.tensor([  0.0000, 0.0000])
                    normState[0] = state[0]/(self.env.T - 1)
                    normState[1] = 10 * (state[1]/(self.env.totalDemand)) - 5   
                    retu = 0

                
                    while not done:
                        discount = discount * self.delta
                        prevState = state
                        normPrevState = normState  
                    
                        probs = self.policy(normPrevState)
                        if probs.max() < 0.95:
                            earlyExit = False
                        if ((episode == self.numberEpiPerBatch - 1) and (batch % 5000 == 0)):
                                print(probs)
                        distAction = Categorical(probs)
                        action = distAction.sample()
                        state, reward, done = self.env.step(prevState, action.item())
                        reward = reward * discount
                        normState = torch.tensor([  0.0000, 0.0000])
                        normState[0] = state[0]/(self.env.T - 1)
                        normState[1] = 10 * (state[1]/(self.env.totalDemand)) - 5
                        retu = retu + reward
                        episodeMemory.append((normPrevState, action, 0))
                   
                        
                    retu -= 5500*len(episodeMemory)
                    retu /= 500
                    states = torch.stack([item[0] for item in episodeMemory])
                    actions = torch.tensor([item[1] for item in episodeMemory])
                    rewards = torch.tensor([retu]*len(episodeMemory))
                    rewards = self.discount_rewards(rewards, self.gamma)
                    batchStates = torch.cat((batchStates,states),0)
                    batchActions = torch.cat((batchActions,actions),0)
                    batchRewards = torch.cat((batchRewards,rewards),0)
                    totalReturn += retu 
                    
                
                if earlyExit:
                    print(batch)
                    break
                action_probs = self.policy(batchStates) 
                prob_batch = action_probs.gather(dim=1,index=batchActions.long().view(-1,1)).squeeze()
                loss = -1 * torch.sum(batchRewards * torch.log(prob_batch))
                x[batch] = loss.item()
                y[batch] = totalReturn
                if batch % 5000 == 0:
                    print(batch, loss.item(), totalReturn)
                    plt.scatter(x[0:batch-1],y[0:batch-1])
                    plt.show()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.returns[iteration][batch] = totalReturn #sum of the our player's rewards  rounds 0-25 

            plt.scatter(x,y)
            plt.show()
                
    
    def discount_rewards(self, rewards, gamma):
        lenr = len(rewards)
        disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards
        return disc_return


    
