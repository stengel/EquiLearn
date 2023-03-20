# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Defines the Q-table

import numpy as np 
import os

class QTable():
    

    def __init__(self, number_demands, number_actions, number_stages, learning_rate):
    
        self.number_demands= number_demands
        self.number_actions = number_actions
        self.number_stages = number_stages
        self.learning_rate = learning_rate
        self.Q_table = np.zeros((self.number_demands, self.number_actions, self.number_actions, self.number_stages))
        self.QTable_name=f"QTable, actions={self.number_actions}, maximum demand={self.number_demands - 1}, stages={self.number_stages}"
        

    def reset(self):
        Qtable = np.zeros((self.number_demands, self.number_actions, self.number_actions, self.number_stages))
        return Qtable, self.learning_rate
    
    
    def random_reset(self):
        random_Qtable = np.random.rand(self.number_demands, self.number_actions, self.number_actions,  self.number_stages)
        return random_Qtable, self.learning_rate
    
    def save(self, name = None):
        if name is None:
            return np.save(os.path.join('QTables',f'{self.QTable_name}'), self.Q_table)
        else:
            return np.save(os.path.join('QTables',name), self.Q_table)
    
    def load(self,name = None):
        if name is None:
            return np.load(os.path.join('QTables',f'{self.QTable_name}'))
        else:
            return np.load(os.path.join('QTables',name))
        
   
    def to_policy_table(self, cost):
        policy_table = np.argmax(self.Q_table, axis = 1)
        for demand_potential in range(self.number_demands):
            conversion_amount = int((demand_potential + cost)/2) - (self.number_actions -1)
            policy_table[demand_potential] += conversion_amount
        return policy_table
        
        


        

