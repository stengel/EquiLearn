import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp


class NNBase():

    """
        Defines a three layer Neural Network that acts as the agent. The output is a tensor of probablities over actions.

        reset() function should be overwritten to define the policy and optimizer
    """

    save_path_format = './NNs/{name}.pt'

    def __init__(self, lr=.000001, num_input=3, num_actions=50, nn_dim=100) -> None:

        self.lr = lr
        self.num_input = num_input
        self.num_actions = num_actions
        self.nn_dim = nn_dim

        self.nn_name = f"nn, lr={self.lr}, actions={self.num_actions}"

    def reset(self):
        self.policy = nn.Sequential(
            nn.Linear(self.num_input, self.nn_dim),
            nn.ReLU(),
            nn.Linear(self.nn_dim, self.nn_dim),
            nn.ReLU(),
            nn.Linear(self.nn_dim, self.num_actions),
            nn.Softmax(dim=0))
        
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        return self.policy, self.optim

    def save(self, name=None):
        self.nn_name = (self.nn_name if name is None else name)
        return torch.save(self.policy.state_dict(), self.save_path_format.format(name=self.nn_name))

    def load(self, name=None):
        self.nn_name = (self.nn_name if name is None else name)
        self.policy.load_state_dict(
            torch.load(self.save_path_format.format(name=self.nn_name)))
        print("\n load NN:",name)


# class PlicyGradient():
#     def __init__(self, num_input=3, num_actions=50, nn_dim=100):
#         return 
    

class ActorCritic(nn.Module): 
    def __init__(self, num_input=3, num_actions=50, nn_dim=100):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(num_input,nn_dim)
        self.l2 = nn.Linear(nn_dim,nn_dim)
        self.actor_lin1 = nn.Linear(nn_dim,num_actions)
        self.l3 = nn.Linear(nn_dim,nn_dim)
        self.critic_lin1 = nn.Linear(nn_dim,1)
    def forward(self,x):
        #x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.softmax(self.actor_lin1(y),dim=0) #C
        c = F.elu(self.l3(y.detach()))
        critic = F.elu(self.critic_lin1(c)) #D
        return actor, critic #E

class NeuralNetwork(NNBase):
    def __init__(self,lr=.000001, num_input=3, num_actions=50, nn_dim=100):
        super().__init__(lr=lr, num_input=num_input, num_actions=num_actions, nn_dim=nn_dim)

    def reset(self):
        self.policy = ActorCritic(num_input=self.num_input, num_actions=self.num_actions, nn_dim=self.nn_dim)
        
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        return self.policy, self.optim
