# Francisco, Sahar, Edward
# ReinforceAlgorithm Class: Solver.

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np # numerical python
import pandas as pd
# printoptions: output limited to 2 digits after decimal point
# np.set_printoptions(precision=2, suppress=False)
import matplotlib.pyplot as plt
import time




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
    
    def run_episode(self, batch, episode):
        if batch % 10 == 0 and (episode == 0):
            print(batch)
        earlyExit = True
        discount = 1 / self.delta                
        episodeMemory = list()
        probMemory = list()
        state, reward, done = self.env.reset()
                
        normState = torch.tensor([  0.0000, 0.0000])
        normState[0] = 2 * (state[0]/(self.env.T - 1)) - 1
        normState[1] = 10 * (state[1]/(self.env.totalDemand)) - 5   
        period = 0
                
        while not done:
            discount = discount * self.delta
            prevState = state
            normPrevState = normState  
                    
            probs = self.policy(normPrevState)
            if (batch % 10 == 0) and (episode == 0) and (period % 6 == 0):
                print(probs)
            distAction = Categorical(probs)
            action = distAction.sample()
            probMemory.append(probs[action])
            if probs[action] < 0.95:
                earlyExit = False
            state, reward, done = self.env.step(prevState, action.item())
            reward = reward * discount
            normState = torch.tensor([  0.0000, 0.0000])
            normState[0] = 2 * (state[0]/(self.env.T - 1)) - 1
            normState[1] = 10 * (state[1]/(self.env.totalDemand)) - 5
            episodeMemory.append((normPrevState, action, reward))
            period += 1
                        

        states = torch.stack([item[0] for item in episodeMemory])
        actions = torch.tensor([item[1] for item in episodeMemory])
        rewards = torch.tensor([item[2] for item in episodeMemory])
        retu = torch.sum(rewards).item()
        futureRewards = self.future_rewards(rewards, self.gamma)
        return states, actions, probMemory, futureRewards, retu, earlyExit
    
    def add_episode_to_batch(self, batchStates, batchActions, batchProbs, batchRewards, batchReturn, 
                             states, actions, probs, rewards, retu):
        batchStates = torch.cat((batchStates,states),0)
        batchActions = torch.cat((batchActions,actions),0)
        batchProbs += probs
        batchRewards = torch.cat((batchRewards,rewards),0)
        batchReturn += retu
        return batchStates, batchActions, batchProbs, batchRewards, batchReturn
    

    def  solver(self):
        """
            Method that performs Monte Carlo Policy Gradient algorithm. 
        """ 
        bestActions = [0] * self.env.T
        bestPayoff = 0
        finalActions = [0] * self.env.T
        finalPayoff = 0
        convergence = 1
        
        for iteration in range(self.numberIterations):
            self.resetPolicyNet()
            returns = list()
            losses = list()         
            
            for batch in range(self.numberBatches):
                batchReturn = 0
                batchStates = torch.empty(0)
                batchActions = torch.empty(0)
                batchProbs = list()
                batchRewards = torch.empty(0)
                st = time.time()
                for episode in range(self.numberEpiPerBatch):
                    states, actions, probMemory, futureRewards, retu, earlyExit = self.run_episode(batch, episode)
                    batchStates, batchActions, batchProbs, batchRewards, batchReturn = \
                    self.add_episode_to_batch(batchStates, batchActions, batchProbs, batchRewards, batchReturn, states, actions, probMemory, futureRewards, retu)
                    
                    if (batch == self.numberBatches - 1) and (episode == self.numberEpiPerBatch - 1):
                        finalActions = actions.numpy()
                        finalPayoff = retu
                        for prob in probMemory:
                            convergence *= prob.item()
                    if retu > bestPayoff:
                        bestActions = actions.numpy()
                        bestPayoff = retu
                    
                ft = time.time()
                print("Episodes", ft-st)
                if earlyExit:
                    print(batch)
                    finalActions = actions.numpy()
                    finalPayoff = retu
                    for item in probMemory:
                        convergence *= item.item()
                    break  
                
                st = time.time()
                action_probs = torch.stack(batchProbs)
                loss = -1 * (batchRewards * torch.log(action_probs)).mean()
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                ft = time.time()
                print("Gradient Descent", ft-st)
                
                losses.append(loss.item())
                batchReturn /= self.numberEpiPerBatch
                returns.append(batchReturn)
                if batch % 50 == 0 and batch > 0:
                    plt.plot(losses)
                    plt.show()
                    plt.plot(returns)
                    plt.show()
                self.returns[iteration][batch] = batchReturn #sum of the our player's rewards  rounds 0-25 
            plt.scatter(losses,returns)
            plt.show()
            plt.plot(losses)
            plt.show()
            return self.returns, finalActions, finalPayoff, convergence, bestActions, bestPayoff


    def future_rewards(self,rewards, gamma):
        lenr = len(rewards)
        futureRewards = torch.Tensor([0]*(lenr))
        total = 0
        for i in range(lenr):
            futureRewards[lenr - i - 1] = (rewards[lenr - i - 1]-6500)/5000 + gamma * total
            total = futureRewards[lenr - i - 1]
        return futureRewards
    
    def discount_total_rewards(self,rewards,gamma):
        lenr = len(rewards)
        total = torch.sum(rewards)
        total -= 35
        total /= 10
        futureRewards = torch.Tensor([total]*(lenr))
        disc_return = torch.pow(gamma,torch.arange(lenr).float()) * futureRewards
        disc_return = torch.flip(disc_return, [0])
        return disc_return
    


    
