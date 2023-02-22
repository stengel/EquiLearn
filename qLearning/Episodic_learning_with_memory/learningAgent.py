# Ed, Kat, Galit
# ReinforceAlgorithm Class: Solver.
# Implement an off-policy Q-learning algorithm

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
        self.number_demands=Qtable.number_demands #Qtable dimen - number of states x number of actions x number of stages
        self.number_stages=Qtable.number_stages
        self.number_actions=Qtable.number_actions
        self.highest_demand = int(self.number_demands - 1) 
        
        
    def reset_Qtable(self):
        self.Qtable, self.learning_rate = self.Qtable.reset()


    def action_index(self, monopoly_price, action):      # computes action index in Qtable
        return int(action - (monopoly_price - self.number_actions + 1)) #monopoly_price should have index number_actions - 1

    def alpha_n(self, n):                       # defines the Qlearning rate
        return self.learning_rate[0]/(n+self.learning_rate[1])
    
    """
    Now the Q-learning itself 
    """
    
    def random_policy(self, monopoly_price):
        random_action = np.random.randint(monopoly_price-self.number_actions+1, monopoly_price + 1)
        if random_action < 0:
            random_action = max(0, random_action)
        if self.env.costs[0] > random_action:
            random_action = monopoly_price 
        return random_action
    
    def q_learning(self, state_action_value, optimal_next_value, reward, alpha, gamma): 
        target = reward + gamma * optimal_next_value
        return state_action_value + alpha * (target - state_action_value)
    
    
    def solver(self):
        
        self.reset_Qtable
        
        for episode in range(self.number_episodes):
            
            if episode % 200_000 == 0:
                print(episode)
            
            state, reward, done = self.env.reset()

            while not done:
                stage, agent_demand, agent_previous_action, adversary_previous_action = state
                if (agent_demand < 0) or (agent_demand > self.highest_demand):
                    break
                if (stage > 0 and adversary_previous_action < self.env.costs[1]) or (self.highest_demand - agent_demand < adversary_previous_action):
                    break
                monopoly_price = int((agent_demand + self.env.costs[0]) / 2) 
                action = self.random_policy(monopoly_price)
                demand_index = int(agent_demand)
                action_index = self.action_index(monopoly_price, action)
                state, reward, done = self.env.step(state, action, action_index)


                # updating the Qtable
                if done: 
                    optimal_next_value = 0
                else:
                    optimal_next_value = max(self.Qtable[int(state[1]),:, action_index, state[0]])
                
                current_q_value = self.Qtable[demand_index, action_index, agent_previous_action, stage]
                self.Qtable[demand_index, action_index, agent_previous_action, stage] = self.q_learning(current_q_value, optimal_next_value, reward, self.alpha_n(episode), self.gamma)

