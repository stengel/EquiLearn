import torch
import torch.nn as nn

class NeuralNetwork():

    save_path_format='./NNs/{name}.pt'

    def __init__(self,lr=0.00001,num_input=3, num_actions=100, nn_dim=100) -> None:
    

        self.lr=lr
        self.num_input= num_input
        self.num_actions = num_actions
        self.nn_ = nn_dim
        self.nn_name=f"nn, lr={self.lr}, actions={self.num_actions}"

        

    def reset(self):
        self.policy = torch.nn.Sequential(
                            nn.Linear(self.num_input, self.nn_), 
                            nn.ReLU(),
                            nn.Linear(self.nn_, self.nn_), 
                            nn.ReLU(),
                            nn.Linear(self.nn_,self.num_actions),                       
                            nn.Softmax(dim=0))
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        return self.policy, self.optim

    def save(self, name=None): 
        self.nn_name=(self.nn_name if name is None else name)
        return torch.save(self.policy.state_dict(),self.save_path_format.format(name=self.nn_name))

    def load(self, name=None):
        self.nn_name=(self.nn_name if name is None else name)
        self.policy=torch.load(self.save_path_format.format(name=self.nn_name))
        
