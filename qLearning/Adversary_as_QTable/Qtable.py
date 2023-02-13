# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Defines the Q-table

import numpy as np  
import os

class QTable():
    

    def __init__(self, num_States, num_Actions, learning_Rate):
    
        self.num_States= num_States
        self.num_Actions = num_Actions
        self.learning_Rate = learning_Rate
        self.Q_table = np.zeros((self.num_States, self.num_Actions))
        self.QTable_name=f"QTable, actions={self.num_Actions}, states={self.num_States}"
        

    def reset(self):
        Qtable = np.zeros((self.num_States, self.num_Actions))
        return Qtable, self.learning_rate
    
    def myopicReset(self):
        Qtable = np.zeros((self.num_States, self.num_Actions))
        for i in range(self.num_States):
            Qtable[i][self.num_Actions-1] = 1
        return Qtable
    
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

    



        

