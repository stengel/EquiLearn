# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Defines the Q-table

import numpy as np  

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
    
    
    def randomReset(self):
        randQtable = np.random.rand(self.num_States,self.num_Actions)
        return randQtable, self.learning_rate


        

