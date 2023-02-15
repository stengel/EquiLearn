# Ed, Kat, Galit
# ReinforceAlgorithm Class: Solver.

import numpy as np #repeated

# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class LearningAlgorithm():
    """
        Model Solver.
    """
    def __init__(self, Model, Qtable, number_episodes, discount_factor) -> None:
        
        self.env = Model
        self.Qtable = Qtable.Q_table
        self.learning_rate = Qtable.learning_rate
        self.number_episodes = number_episodes
        self.gamma = discount_factor
        self.number_states=Qtable.number_states #Qtable dimen - number of states x number of actions x number of stages
        self.number_stages=Qtable.number_stages
        if self.number_states % 2 ==1 :
            print('num of states should be even')
        self.number_actions=Qtable.number_actions
        if self.number_actions % 2 ==1 :
            print('num of actions should be even')
        self.lowest_state = int(200-(self.number_states)/2) # Should already be int
        self.highest_state = int(200+(self.number_states)/2 - 1) # Should already be int
        
        
    def reset_Qtable(self):
        self.Qtable, self.learning_rate = self.Qtable.reset()
        """
        This function will eventually choose and adversary agent to play against
        and according to that return an action played by that adversary. 
        This could be chosen according to some existing equilibruim dist.
        """
    def choose_adversary(self, state):
        return int((400 - state + self.env.costs[1])/2) #myopic

    def state_index(self, state):                 # computes state index in Qtable
        return int(state - self.lowest_state) 
        #return min(int(state -(200-self.number_states/2)), self.number_states - 1)

    def action_index(self, monopoly_price, action):      # computes action index in Qtable
        return int(action - (monopoly_price - self.number_actions + 1)) #monopoly_price should have index number_actions - 1

    def alpha_n(self, n):                       # defines the Qlearning rate
        return self.learning_rate[0]/(n+self.learning_rate[1])
    """
    Now the Q-learning itself 
    """
    def  solver(self):
        
        self.reset_Qtable

        for episode in range(self.number_episodes):

            state = np.random.randint(self.lowest_state, self.highest_state + 1) 
            stage = np.random.randint(0, self.number_stages)
            monopoly_price = int((state + self.env.costs[0]) / 2) 
            action = np.random.randint(monopoly_price-self.number_actions+1, monopoly_price + 1)
            adversary_action = self.choose_adversary(state)
            reward = (state-action)*(action-self.env.costs[0])
            next_state = int(state+.5*(adversary_action-action))
            if(stage==self.number_stages-1): 
                optimal_next_value=0
            else:
                optimal_next_value=max(self.Qtable[self.state_index(next_state),:, stage+1])
        

            # updating the Qtable
            q_value = (1-self.alpha_n(episode)) * \
                   self.Qtable[self.state_index(state), self.action_index(monopoly_price, action), stage]\
                   + self.alpha_n(episode) * \
                   ((1-self.gamma)*reward + self.gamma*optimal_next_value)
            self.Qtable[self.state_index(state), self.action_index(monopoly_price, action), stage]=q_value



