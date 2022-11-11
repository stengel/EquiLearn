import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class DemandPotentialGame():
    """
        Fully defines demand Potential Game. 
    """
    
    def __init__(self, totalDemand, tupleCosts, adversaryPolicy, stages) -> None:
        self.totalDemand = totalDemand
        self.tupleCosts = tupleCosts
        self.stages = stages
        self.adversaryPolicy = adversaryPolicy
        self.adversaryReturns = None
        self.adpersaryPrices = list()



    def profits(self, demand, p1, p2):
        return None

    def demandAssignment(previousDemand, p1, p2):
        return None

    

class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined.
    """
    def __init__(self, totalDemand, tupleCosts, adversaryPolicy, transitionFunction, actionSpace, initState) -> None:
        super().__init__(totalDemand, tupleCosts, adversaryPolicy)
        self.transitionFunction = transitionFunction
        self.actionSpace = actionSpace
        self.rewardFunction = self.profits
        self.stage = None
        self.state = None
        self.initState = initState
        self.done = False

    def reset(self):
        reward = 0
        self.stage = 0
        self.state = self.initState
        return self.state, reward, self.done




class REINFORCEALgorithm():
    """
        Model Solver
    """
    def __init__(self, Model, policyNet, numberEpisodes) -> None:
        self.env = Model
        self.environment.adversaryReturns = np.zeros(numberEpisodes)
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

                state, reward, done, _ = env.step(action.item())
                
                episodeMemory.append((prev_state, action, reward))
                self.stage += 1

        states = torch.stack([item[0] for item in episodeMemory])
        actions = torch.tensor([item[1] for item in episodeMemory])
        rewards = torch.tensor([item[2] for item in episodeMemory])

        action_probs = policy(states)
        action_dists = Categorical(action_probs)
        action_logprobs = action_dists.log_prob(actions)

        returns = self.returns(reward, episodeMemory)

        loss = - ( torch.sum(returns*action_logprobs) )/len(episodeMemory)

        optim.sero_grad()
        loss.backward()
        optim.step()

    def returns(self, reward, episodeMemory):
        return torch.tensor( [torch.sum( reward[i:]*
        (gamma**torch.arange(i, len(episodeMemory) )) ) for in range(len(episodeMemory))] )
	 
