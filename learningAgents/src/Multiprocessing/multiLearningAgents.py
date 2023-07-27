import torch
import torch.nn as nn
from torch.distributions import Categorical
from Environment import Model
import numpy as np # numerical python
import pandas as pd
# printoptions: output limited to 2 digits after decimal point
# np.set_printoptions(precision=2, suppress=False)
import matplotlib.pyplot as plt
import multiprocessing as mp



class Solver():

    
    def __init__(self,numberEpiPerBatch, Model, discountFactor, creditFactor, numberIterations, numberBatches):
        self.numberEpiPerBatch = int(numberEpiPerBatch)    
        self.env = Model
        self.delta = discountFactor # Defined by how patient the player is
        self.gamma = creditFactor # Defined by how much credit we give previous actions
        self.numberIterations = int(numberIterations)
        self.numberBatches = int(numberBatches)
        
        

class ReinforceAlgorithm(Solver):
    """
        Model Solver.
    """
    def __init__(self, Model, neuralNet, numberIterations, numberEpiPerBatch, numberBatches, discountFactor, creditFactor) -> None:
        super().__init__(numberEpiPerBatch, Model, discountFactor, creditFactor, numberIterations, numberBatches)

        self.neuralNetwork = neuralNet  
        self.policy = None
        self.optim = None
        self.returns = np.zeros((int(numberIterations), int(numberBatches))) 


    def resetPolicyNet(self):
        """
            Reset Policy Neural Network.
        """
        self.policy, self.optim = self.neuralNetwork.reset()
        
    
    def run_episode(self, t):
        discount = 1 / self.delta                
        episodeMemory = list()
        probMemory = list()
        state, reward, done = self.env.reset()
                
        normState = torch.tensor([  0.0000, 0.0000])
        normState[0] = 2 * (state[0]/(self.env.T - 1)) - 1
        normState[1] = 10 * (state[1]/(self.env.totalDemand)) - 5   
        retu = 0
        period = 0
                
        while not done:
            discount = discount * self.delta
            prevState = state
            normPrevState = normState  
                    
            probs = self.policy(normPrevState)
            distAction = Categorical(probs)
            action = distAction.sample()
            probMemory.append(probs[action])
            state, reward, done = self.env.step(prevState, action.item())
            reward = reward * discount
            retu += reward
            reward -= 5500
            reward /= 1000
            normState = torch.tensor([  0.0000, 0.0000])
            normState[0] = 2 * (state[0]/(self.env.T - 1)) - 1
            normState[1] = 10 * (state[1]/(self.env.totalDemand)) - 5
            episodeMemory.append((normPrevState, action, reward))
            period += 1
                        

        states = torch.stack([item[0] for item in episodeMemory])
        actions = torch.tensor([item[1] for item in episodeMemory])
        rewards = torch.tensor([item[2] for item in episodeMemory])
        futureRewards = self.discount_total_rewards(rewards, self.gamma)
        return states, actions, probMemory, futureRewards, retu
    

    def  solver(self):
        """
            Method that performs Monte Carlo Policy Gradient algorithm. 
        """ 
#         bestActions = [0] * self.env.T
#         bestPayoff = 0
#         finalActions = [0] * self.env.T
#         finalPayoff = 0
#         convergence = 1
        
        
        for iteration in range(self.numberIterations):
            self.resetPolicyNet()        
            
            for batch in range(self.numberBatches):
                totalReturn = 0
                batchStates = torch.empty(0)
                batchActions = torch.empty(0)
                batchProbs = list()
                batchRewards = torch.empty(0)
                
                
                PROCESSES = 4
                ctx = mp.get_context('spawn')
                with ctx.Pool(PROCESSES) as pool:
                    params = [(0, )]*self.numberEpiPerBatch
                    results = [pool.apply_async(self.run_episode, p) for p in params]
                    for r in results:
                        print('\t', r.get())
                
                for episode in range(self.numberEpiPerBatch):
                    states, actions, probMemory, futureRewards, retu = self.run_episode()
#                     if (batch == self.numberBatches - 1) and (episode == self.numberEpiPerBatch - 1):
#                         finalActions = actions.numpy()
#                         finalPayoff = retu
#                         for item in probMemory:
#                             convergence *= item.item()
#                     if retu > bestPayoff:
#                         bestActions = actions.numpy()
#                         bestPayoff = retu
                    totalReturn += retu
                    batchStates = torch.cat((batchStates,states),0)
                    batchActions = torch.cat((batchActions,actions),0)
                    batchProbs += probMemory
                    batchRewards = torch.cat((batchRewards,futureRewards),0)    
                    
                action_probs = torch.stack(batchProbs)
                loss = -1 * (batchRewards * torch.log(action_probs)).mean()
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                
                totalReturn /= self.numberEpiPerBatch
                self.returns[iteration][batch] = totalReturn #sum of the our player's rewards  rounds 0-25 
#         return self.returns, finalActions, finalPayoff, convergence, bestActions, bestPayoff
    

    def future_rewards(self,rewards, gamma):
        lenr = len(rewards)
        futureRewards = torch.Tensor([0]*(lenr))
        total = 0
        for i in range(lenr):
            futureRewards[lenr - i - 1] = rewards[lenr - i - 1] + gamma * total
            total = futureRewards[lenr - i - 1]
#         futureRewards -= futureRewards.mean()
#         futureRewards /= futureRewards.max()
#         futureRewards /= futureRewards.std().clamp_min(1e-12)
        return futureRewards
    
    def discount_total_rewards(self,rewards,gamma):
        lenr = len(rewards)
        total = torch.sum(rewards)
        total -= 3.5
        futureRewards = torch.Tensor([total]*(lenr))
        disc_return = torch.pow(gamma,torch.arange(lenr).float()) * futureRewards
        disc_return = torch.flip(disc_return, [0])
        return disc_return
    


    

