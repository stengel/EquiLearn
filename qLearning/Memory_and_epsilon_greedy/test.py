# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Testing out the Q-table against a given opponent

import numpy as np
from environment import AdversaryModes
import matplotlib.pyplot as plt


class Test():
    

    def __init__(self, Model, Qtable, discount_factor, adversary_probabilities) -> None:

        
        self.env = Model
        self.Qtable = Qtable.Q_table 
        self.number_demands=Qtable.number_demands
        self.highest_demand = self.number_demands - 1
        self.number_actions=Qtable.number_actions
        self.number_stages=Qtable.number_stages
        self.discount_factor = discount_factor  
        self.adversary_probabilities = adversary_probabilities
        self.adversary = None
        
    
    
    def set_adversary(self):
        options = list(range(len(self.adversary_probabilities)))
        adversary_index = int(np.random.choice(options, 1, p= self.adversary_probabilities))
        self.adversary = AdversaryModes(adversary_index)
        new_probabilities = [0]*len(self.adversary_probabilities)
        new_probabilities[adversary_index] = 1
        self.env.adversary_probabilities = new_probabilities
            

    def total_payoff(self):

        self.set_adversary()
        
        delta = 1
        utility = 0
        adversary_utility = 0
        actions = [0]*self.number_stages
        adversary_actions = [0]*self.number_stages
        demands = [0]*self.number_stages
        state_vector, reward, done = self.env.reset()
        demand = state_vector[1]
        previous_action = state_vector[2]
        
        for stage in range(self.number_stages):
            demands[stage] = demand
            if demand >= self.number_demands:
                print("max demand reached")
                demand = self.number_demands - 1
            if demand < 0:
                print("min demand reached")
                demand = 0
            demand_index = int(demand)
            action_index = np.argmax(self.Qtable[demand_index, :, previous_action, stage])
            action = action_index + int((demand + self.env.costs[0])/2) - self.number_actions + 1
            actions[stage] = action
            utility += (demand-action)*(action-self.env.costs[0]) * delta
            adversary_demand = 400 - demand
            state_vector, _, _ = self.env.step(state_vector, action, action_index)
            demand = state_vector[1]
            previous_action = state_vector[2]
            adversary_actions[stage] = state_vector[3]
            adversary_utility += (adversary_demand-adversary_actions[stage])*(adversary_actions[stage]-self.env.costs[1]) * delta
            delta *= self.discount_factor
        return utility, adversary_utility, np.transpose(actions), np.transpose(adversary_actions), np.transpose(demands)


    def q_learning(self, state_action_value, optimal_next_value, reward, alpha, gamma): 
        target = reward + gamma * optimal_next_value
        return state_action_value + alpha * (target - state_action_value)
    
    def random_policy(self, monopoly_price):
        random_action = np.random.randint(monopoly_price-self.number_actions+1, monopoly_price + 1)
        if random_action < 0:
            random_action = max(0, random_action)
        if self.env.costs[0] > random_action:
            random_action = monopoly_price 
        return random_action
    
    def action_index(self, monopoly_price, action):      # computes action index in Qtable
        return int(action - (monopoly_price - self.number_actions + 1)) #monopoly_price should have index number_actions - 1
    
    
    def error(self, number_tests):
        

        
        errors = list()
        
        for test in range(number_tests):

            self.set_adversary()
            
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

                if done: 
                    optimal_next_value = 0
                else:
                    optimal_next_value = max(self.Qtable[int(state[1]),:, action_index, state[0]])
                    
                new_value = reward + self.discount_factor * optimal_next_value
                current_q_value = self.Qtable[demand_index, action_index, agent_previous_action, stage]
                
                if new_value != 0:
                    errors.append((new_value - current_q_value)/new_value)
                    
#                 if new_value == 0 and current_q_value != 0:
#                     print("Division by zero error") 
                    
        error_array = np.array(errors)            
#         plt.plot(error_array)
        return error_array.mean()
    
        

    
#     def myopic(self, cost, demand):
#         return (cost + demand)/2
    
#     def payoff(self, cost, demand, price):
#         return (demand - price)*(price - cost)
       
#     def update_demand(self, demand, price_pair):
#         new_demand = demand + 0.5*(price_pair[1]- price_pair[0])
#         return new_demand
    
#     def utility_of_actions(self, actions):
#         agent_demand = 200
#         opponent_demand = 200
#         total_payoff = 0
#         delta = 1/self.discount_factor
#         for i in range(self.number_stages):
#             delta = delta * self.discount_factor
#             agent_price = int(actions[i])
#             opponent_price = int(self.myopic(self.env.costs[1],opponent_demand))
#             total_payoff += self.payoff(self.env.costs[0],agent_demand,agent_price) * delta
#             agent_demand = int(self.update_demand(agent_demand, [agent_price,opponent_price]))
#             opponent_demand = 400 - agent_demand
#         return total_payoff
            

    


                



        

