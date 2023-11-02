import torch
import torch.nn as nn
from torch.nn import functional as F
import time


class NNBase():

    """
        Defines a one layer Neural Network that acts as the agent. The output is a tensor of probablities over actions.

        reset() function defines the policy and optimizer
    """

    SAVE_PATH_FORMAT = './NNs/{name}.pt'
    policy=None
    optim=None
    DIMENSION=512

    def __init__(self, lr, num_input, num_actions) -> None:

        self.lr = lr
        self.num_input = num_input
        self.num_actions = num_actions
        

        self.name = f"nn, lr={self.lr}, numActions={self.num_actions},{int(time.time())}"

    def reset(self):
        
        self.policy = nn.Sequential(
            nn.Linear(self.num_input, self.DIMENSION),
            nn.ReLU(),
            # nn.Linear(self.nn_dim, self.nn_dim),
            # nn.ReLU(),
            nn.Linear(self.DIMENSION, self.num_actions),
            nn.Softmax(dim=0))
        
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        # print(self.policy[0].weight)
        # print(self.policy[2].weight)
        # print("policy reset")
        return self.policy, self.optim

    def save(self, name=None):
        self.name = (self.name if name is None else name)
        # print("policy saved!")
        return torch.save(self.policy.state_dict(), self.SAVE_PATH_FORMAT.format(name=self.name))

    def load(self, name=None):
        if self.policy is None:
            self.reset()
        self.name = (self.name if name is None else name)
        self.policy.load_state_dict(
            torch.load(self.SAVE_PATH_FORMAT.format(name=self.name)))
        # print("policy loaded!")


