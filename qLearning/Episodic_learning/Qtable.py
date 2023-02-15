# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Defines the Q-table

import numpy as np  

class QTable():
    

    def __init__(self, number_states, number_actions, number_stages, learning_rate):
    
        self.number_states= number_states
        self.number_actions = number_actions
        self.number_stages = number_stages
        self.learning_rate = learning_rate
        self.Q_table = np.zeros((self.number_states, self.number_actions, self.number_stages))
        self.QTable_name=f"QTable, actions={self.number_actions}, states={self.number_states}, stages={self.number_stages}"
        

    def reset(self):
        Qtable = np.zeros((self.number_states, self.number_actions, self.number_stages))
        return Qtable, self.learning_rate
    
    
    def random_reset(self):
        random_Qtable = np.random.rand(self.number_states,self.number_actions, self.number_stages)
        return random_Qtable, self.learning_rate


        

