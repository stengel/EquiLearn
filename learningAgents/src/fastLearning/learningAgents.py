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

    
    def __init__(self,numberEpiPerBatch, Model, numberBatches):
        self.numberEpiPerBatch = numberEpiPerBatch    
        self.env = Model
        self.numberBatches = numberBatches
        
        

class ReinforceAlgorithm(Solver):
    """
        Model Solver.
    """
    def __init__(self, Model, neuralNet, numberEpiPerBatch, numberBatches) -> None:
        super().__init__(numberEpiPerBatch, Model, numberBatches)

        self.neuralNetwork = neuralNet  
        self.policy = None
        self.optim = None
        self.returns = np.zeros(numberBatches) 


    def resetPolicyNet(self):
        """
            Reset Policy Neural Network.
        """
        self.policy, self.optim = self.neuralNetwork.reset()
    
    def normalise_state(self, state):
        return torch.tensor([2 * (state[0]/(self.env.T - 1)) - 1, \
                             10 * (state[1]/(self.env.totalDemand)) - 5])
    
    
    
    def run_episode(self, batch, episode):
        PRINT_EVERY = 50
        episodeMemory = list()
        state, reward, done = self.env.reset()
        if batch % PRINT_EVERY == 0 and episode == 0:
            print_memory = list()
                
        normState = self.normalise_state(state)
                
        while not done:
            prevState = state
            normPrevState = normState  
            probs = self.policy(normPrevState)
            if batch % PRINT_EVERY == 0 and episode == 0 and len(episodeMemory) % 12 == 0:
                print(probs)
            distAction = Categorical(probs)
            action = distAction.sample()
            if batch % PRINT_EVERY == 0 and episode == 0:
                print_memory.append(action.item())
            probability = probs[action]
            state, reward, done = self.env.step(prevState, action.item())
            normState = self.normalise_state(state)
            episodeMemory.append((reward, probability, action))

        rewards = torch.tensor([item[0] for item in episodeMemory])
        probabilities = torch.stack([item[1] for item in episodeMemory])
        actions = torch.tensor([item[2] for item in episodeMemory])
        retu = torch.sum(rewards).item()
        futureRewards = self.future_rewards(rewards)
        if batch % PRINT_EVERY == 0 and episode == 0:
            print("Actions:", print_memory, "Return:", retu)
        return probabilities, futureRewards, retu, actions
    

    def  solver(self):
        """
            Method that performs Policy Gradient algorithm. 
        """ 
        self.resetPolicyNet() 
        baseline = torch.tensor([0] * self.env.T)
        best_actions = torch.tensor([0] * self.env.T)
        best_payoff = 0
        
            
        for batch in range(self.numberBatches):
            episodeResults = [self.run_episode(batch, episode) for episode in range(self.numberEpiPerBatch)]

            retus = [result[2] for result in episodeResults]
            if max(retus) > best_payoff:
                best_payoff = max(retus)
                best_actions = episodeResults[np.argmax(retus)][3]
                baseline = episodeResults[np.argmax(retus)][1]
                print("Best payoff:", best_payoff, "Best actions:", best_actions)
            probs = [result[0] for result in episodeResults]
            rewards = [result[1] - baseline for result in episodeResults]
            # rewards = [result[1] - baseline/max(batch - 1,1) for result in episodeResults]
            # baseline = rewards[0] + baseline
            batchProbs = torch.cat(probs, 0)
            batchRewards = torch.cat(rewards, 0)
            batchRewards -= batchRewards.mean()
            batchRewards /= batchRewards.std().clamp_min(1e-12)
            batchReturn = np.sum(retus)/self.numberEpiPerBatch
                
            loss = -1 * (batchRewards * torch.log(batchProbs)).mean()
                
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if batch % 10 ==0:
                print("Batch:", batch, "Batch return:", batchReturn)   
            self.returns[batch] = batchReturn # average sum of the our player's rewards rounds 0-25 
            if batchProbs.mean() > 0.97:
                print("Final batch:", batch)
                break
        _, _, retu, actions = self.run_episode(1, 1)
        print("Final payoff:", retu, "Final actions:", actions)
        print("Final best payoff:", best_payoff, "Final best actions:", best_actions)

    def future_rewards(self,rewards):
        rewards = (rewards-6500)/5000
        rewards = torch.flip(rewards, [0])
        rewards = torch.cumsum(rewards, 0)
        rewards = torch.flip(rewards, [0])
        return rewards
    


    
