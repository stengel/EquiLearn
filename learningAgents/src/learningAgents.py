import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import numpy as np # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)





class ReinforceAlgorithm():
    """
        Model Solver.
    """
    def __init__(self, Model, policyNet, optim, numberEpisodes) -> None:
        self.env = Model
        self.env.adversaryReturns = np.zeros(numberEpisodes)
        self.returns = np.zeros(numberEpisodes)
        self.policy = policyNet
        self.numberEpisodes = numberEpisodes
        self.optim = optim
        self.episodesMemory = list()

    def  solver(self):

        for episode in range(self.numberEpisodes):
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
            print(loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.returns[episode] = retu



    def returnsComputation(self, rewards, episodeMemory):
        gamma = .9
        return torch.tensor( [torch.sum( rewards[i:] * (gamma ** torch.arange(i, len(episodeMemory))) ) for i in range(len(episodeMemory)) ] )
	 
