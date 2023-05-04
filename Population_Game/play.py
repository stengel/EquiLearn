# Relates to the Q-Learning approach.
# playing strategy tables against each other

import numpy as np



class Tournament():

    def __init__(self, low_cost_strategies, high_cost_strategies, discount_factor, costs, number_rounds, initial_demands):
        self.high_cost_strategies = high_cost_strategies 
        self.low_cost_strategies = low_cost_strategies
        self.discount_factor = discount_factor  
        self.costs = costs
        self.number_rounds = number_rounds
        self.initial_demands = initial_demands
        
    def run_tournament(self):
        low_cost_payoffs = np.zeros((len(self.low_cost_strategies),len(self.high_cost_strategies)))
        high_cost_payoffs = np.zeros((len(self.low_cost_strategies),len(self.high_cost_strategies)))
        for low_index, low_cost_strategy in enumerate(self.low_cost_strategies):
            for high_index, high_cost_strategy in enumerate(self.high_cost_strategies):
                low_cost_payoffs[low_index][high_index], high_cost_payoffs[low_index][high_index] = self.match(low_cost_strategy, high_cost_strategy)
        return low_cost_payoffs, high_cost_payoffs
    
    def match(self, low_cost_strategy, high_cost_strategy):
        delta = 1
        low_utility = 0
        high_utility = 0
        low_demand, high_demand = self.initial_demands
        low_previous_action_index = 0
        high_previous_action_index = 0
        low_number_actions = len(low_cost_strategy[int(low_demand), :, 0])
        high_number_actions = len(high_cost_strategy[int(high_demand), :, 0])
        number_actions = [low_number_actions, high_number_actions]
        for round_ in range(self.number_rounds):
            low_demand = int(low_demand)
            high_demand = int(high_demand)
            low_action = low_cost_strategy[low_demand][low_previous_action_index][round_]
            high_action = high_cost_strategy[high_demand][high_previous_action_index][round_]
            low_demand, high_demand, low_reward, high_reward, low_previous_action_index, high_previous_action_index = self.update_demands_rewards_prices(low_demand, high_demand, low_action, high_action, self.costs, number_actions)
            low_utility += (low_reward * delta)
            high_utility += (high_reward * delta)
            delta *= self.discount_factor
        return [(low_utility / self.number_rounds), (high_utility / self.number_rounds)]
    
    def return_trajectory(self, low_cost_strategy, high_cost_strategy):
        trajectory = list()
        # delta = 1
        # low_utility = 0
        # high_utility = 0
        low_demand, high_demand = self.initial_demands
        low_previous_action_index = 0
        high_previous_action_index = 0
        low_number_actions = len(low_cost_strategy[int(low_demand), :, 0])
        high_number_actions = len(high_cost_strategy[int(high_demand), :, 0])
        number_actions = [low_number_actions, high_number_actions]
        for round_ in range(self.number_rounds):
            low_demand = int(low_demand)
            high_demand = int(high_demand)
            trajectory.append((low_demand, high_demand))
            low_action = low_cost_strategy[low_demand][low_previous_action_index][round_]
            high_action = high_cost_strategy[high_demand][high_previous_action_index][round_]
            low_demand, high_demand, low_reward, high_reward, low_previous_action_index, high_previous_action_index = self.update_demands_rewards_prices(low_demand, high_demand, low_action, high_action, self.costs, number_actions)
            # low_utility += (low_reward * delta)
            # high_utility += (high_reward * delta)
            # delta *= self.discount_factor
        return trajectory
        
        
    def update_demands_rewards_prices(self, low_demand, high_demand, low_action, high_action, costs, number_actions):
        low_reward = (low_demand - low_action) * (low_action - costs[0])
        high_reward = (high_demand - high_action) * (high_action - costs[1])
        low_previous_action_index = number_actions[0] - 1 - ((int((low_demand + costs[0])/2)) - low_action)
        high_previous_action_index = number_actions[1] - 1 - ((int((high_demand + costs[1])/2)) - high_action)
        low_demand += 0.5 * (high_action - low_action)
        low_demand = int(low_demand)
        high_demand = int(self.initial_demands[0] + self.initial_demands[1] - low_demand)
        return low_demand, high_demand, low_reward, high_reward, low_previous_action_index, high_previous_action_index
    

