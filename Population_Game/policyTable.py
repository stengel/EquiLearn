# Defines a Policy Table- similar to QTable, but only with suggested actions

import numpy as np 
import os

class PolicyTable():
    

    def __init__(self, number_demands, number_actions, cost, number_stages):
    
        self.number_demands= number_demands
        self.number_actions = number_actions
        self.number_stages = number_stages
        self.cost = cost
        self.policy_table = np.zeros((self.number_demands, self.number_actions, self.number_stages))
        self.policy_table_name=f"Policy table, actions={self.number_actions}, cost={self.cost}"
        

    def reset(self):
        self.policy_table = np.zeros((self.number_demands, self.number_actions, self.number_stages))    
    
    def random_reset(self):
        adversary = np.random.randint(0, self.number_actions, (self.number_demands, self.number_actions,  self.number_stages))
        for demand_potential in range(self.number_demands):         
            conversion_amount = int((demand_potential + self.cost)/2) - (self.number_actions -1)
            adversary[demand_potential] += conversion_amount
        return adversary
    
    def save(self, name = None):
        if name is None:
            return np.save(os.path.join('PolicyTables',f'{self.policy_table_name}'), self.policy_table)
        else:
            return np.save(os.path.join('PolicyTables',name), self.policy_table)
    
    def load(self,name = None):
        if name is None:
            return np.load(os.path.join('PolicyTables',f'{self.policy_table_name}'))
        else:
            return np.load(os.path.join('PolicyTables',name))
        
    def define(self, policy_table):
        self.policy_table = policy_table
        


        

