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
        self.number_demands=Qtable.number_demands
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
        best_payoff = 0
        best_actions = list()
        
        for episode in range(self.number_episodes):
            
            payoff = 0
            actions = list()
            discount = 1
            state, reward, done = self.env.reset()

            while not done:
                stage, agent_demand, agent_previous_action, adversary_previous_action = state
                if (agent_demand < 0) or (agent_demand > self.highest_demand):
                    break
                if (stage > 0 and adversary_previous_action < self.env.costs[1]) or (self.highest_demand - agent_demand < adversary_previous_action):
                    break
                monopoly_price = int((agent_demand + self.env.costs[0]) / 2) 
                action = self.random_policy(monopoly_price)
                actions.append(action)
                demand_index = int(agent_demand)
                action_index = self.action_index(monopoly_price, action)
                state, reward, done = self.env.step(state, action, action_index)
                payoff += reward * discount
                discount *= self.gamma


                # updating the Qtable
                if done: 
                    optimal_next_value = 0
                else:
                    optimal_next_value = max(self.Qtable[int(state[1]),:, action_index, state[0]])
                
                current_q_value = self.Qtable[demand_index, action_index, agent_previous_action, stage]
                self.Qtable[demand_index, action_index, agent_previous_action, stage] = self.q_learning(current_q_value, optimal_next_value, reward, self.alpha_n(episode), self.gamma)
                
            if payoff > best_payoff:
                best_payoff = payoff
                best_actions = actions
            if episode == self.number_episodes - 1:
                print("Best payoff: ", best_payoff)
                print("Best actions: ", best_actions)
                    
                    
    def continue_learning(self, number_episodes, number_previous_episodes):
        
        for episode in range(number_episodes):
            
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
                self.Qtable[demand_index, action_index, agent_previous_action, stage] = self.q_learning(current_q_value, optimal_next_value, reward, self.alpha_n(episode + number_previous_episodes), self.gamma)
                
                
    def epsilon_greedy_learning(self, number_episodes, number_previous_episodes):
        
        best_payoff = 0
        best_actions = list()
        final_prob = 0.05
        constant = (final_prob*number_episodes)/(1-final_prob)
        
        for episode in range(number_episodes):
            
            state, reward, done = self.env.reset()
            
            payoff = 0
            actions = list()
            discount = 1

            while not done:
                stage, agent_demand, agent_previous_action, adversary_previous_action = state
                if (agent_demand < 0) or (agent_demand > self.highest_demand):
                    break
                if (stage > 0 and adversary_previous_action < self.env.costs[1]) or (self.highest_demand - agent_demand < adversary_previous_action):
                    break
                demand_index = int(agent_demand)
                monopoly_price = int((agent_demand + self.env.costs[0]) / 2)
                greedy_action_index = np.argmax(self.Qtable[demand_index, :, agent_previous_action, stage])
                action = self.epsilon_greedy_policy(monopoly_price, episode, number_previous_episodes, greedy_action_index, constant)
                actions.append(action)
                action_index = self.action_index(monopoly_price, action)
                state, reward, done = self.env.step(state, action, action_index)
                payoff += reward * discount
                discount *= self.gamma

                # updating the Qtable
                if done: 
                    optimal_next_value = 0
                else:
                    optimal_next_value = max(self.Qtable[int(state[1]),:, action_index, state[0]])
                
                current_q_value = self.Qtable[demand_index, action_index, agent_previous_action, stage]
                self.Qtable[demand_index, action_index, agent_previous_action, stage] = self.q_learning(current_q_value, optimal_next_value, reward, self.alpha_n(episode), self.gamma)
#                 if stage == 0 and current_q_value > 95000:
#                     print("old", current_q_value)
#                     print("new", self.Qtable[demand_index, action_index, agent_previous_action, stage])
#                     print(np.max(self.Qtable))
                    
        
            if payoff > best_payoff:
                best_payoff = payoff
                best_actions = actions
            if episode == number_episodes - 1:
                print("Best payoff: ", best_payoff)
                print("Best actions: ", best_actions)
                
                
    def epsilon_greedy_policy(self, monopoly_price, episode, number_previous_episodes, greedy_action_index, constant):
        epsilon = (constant/(constant + episode)) * (1-(number_previous_episodes /self.number_episodes))
        if np.random.binomial(1,epsilon) == 1:
            action = np.random.randint(monopoly_price-self.number_actions+1, monopoly_price + 1)
        else:
            action = greedy_action_index + (monopoly_price - self.number_actions + 1)
        action = max(0, action)
        if self.env.costs[0] > action:
            action = monopoly_price 
        return action
                
        

